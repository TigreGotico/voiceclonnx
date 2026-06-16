"""Export FreeVC components (WavLM-Large full encoder + SynthesizerTrn decoder) to ONNX.

FreeVC architecture (Qian et al., ICASSP 2023):
  1. WavLM-Large encoder — FULL final hidden states (NOT a specific layer).
     Unlike kNN-VC which uses layer 6, FreeVC calls extract_features()[0]
     which returns the transformer's final output, shape (batch, time, 1024).
  2. Speaker encoder — GE2E LSTM+linear, produces 256-dim d-vector from mel.
  3. SynthesizerTrn (VITS decoder) — prior encoder (enc_p) + flow + Generator
     conditioned on speaker d-vector; inference takes (c, g) → waveform.

WavLM reuse note:
  kNN-VC exports wavlm_layer6.onnx (layer-6 intermediate hidden states).
  FreeVC uses the full-model final output — a DIFFERENT extraction.
  A separate wavlm_freevc.onnx is required; the kNN-VC artifact CANNOT be
  reused.  Both artifacts cross-reference from their respective adapters.

Upstream:
  - https://github.com/OlaWod/FreeVC  (MIT, Qian et al. 2023)
  - WavLM-Large weights: microsoft/wavlm-large on HF (MIT)
  - Checkpoint: OpenVINO notebooks model mirror (MIT)
  - Speaker encoder: pretrained_bak_5805000.pt from HF spaces (MIT)

Usage::

    python -m conversion.export_freevc --output-dir /tmp/freevc-out

    # Or with staging dir and no HF push
    python -m conversion.export_freevc --output-dir /tmp/freevc-out --no-push
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WAVLM_HF_MODEL = "microsoft/wavlm-large"
FREEVC_CHECKPOINT_URL = (
    "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/freevc/freevc.pth"
)
SPEAKER_ENCODER_URL = (
    "https://huggingface.co/spaces/OlaWod/FreeVC/resolve/main/"
    "speaker_encoder/ckpt/pretrained_bak_5805000.pt"
)
FREEVC_UPSTREAM_URL = "https://github.com/OlaWod/FreeVC"
FREEVC_UPSTREAM_REF = "master"  # no versioned release tag exists

# FreeVC freevc.json model config values
_SPEC_CHANNELS = 641          # filter_length // 2 + 1 = 1280 // 2 + 1
_SEGMENT_SIZE = 28            # segment_size // hop_length = 8960 // 320
_SAMPLING_RATE = 16000
_HOP_LENGTH = 320

MIT_LICENSE = """\
MIT License

Copyright (c) 2023 Qian et al. (FreeVC authors)
Copyright (c) 2021 Microsoft Corporation (WavLM-Large weights)
Copyright (c) 2019 Resemblyzer Authors (speaker encoder)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


# ---------------------------------------------------------------------------
# Architecture reconstruction (no FreeVC repo checkout needed)
# All modules are self-contained below; identical to OlaWod/FreeVC master.
# ---------------------------------------------------------------------------

def _build_architecture():
    """Return (commons, modules_ns) namespace objects matching FreeVC source."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import Conv1d
    from torch.nn.utils import weight_norm, remove_weight_norm

    # --- commons helpers ---
    class _Commons:
        @staticmethod
        def sequence_mask(length, max_length=None):
            if max_length is None:
                max_length = length.max()
            x = torch.arange(max_length, dtype=length.dtype, device=length.device)
            return x.unsqueeze(0) < length.unsqueeze(1)

        @staticmethod
        def init_weights(m, mean=0.0, std=0.01):
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                m.weight.data.normal_(mean, std)

        @staticmethod
        def get_padding(kernel_size, dilation=1):
            return int((kernel_size * dilation - dilation) / 2)

    commons = _Commons()

    # --- modules ---
    LRELU_SLOPE = 0.1

    class ResBlock1(nn.Module):
        def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
            super().__init__()
            self.convs1 = nn.ModuleList([
                weight_norm(Conv1d(channels, channels, kernel_size, 1,
                    dilation=d, padding=commons.get_padding(kernel_size, d)))
                for d in dilation
            ])
            self.convs2 = nn.ModuleList([
                weight_norm(Conv1d(channels, channels, kernel_size, 1,
                    dilation=1, padding=commons.get_padding(kernel_size, 1)))
                for _ in dilation
            ])

        def forward(self, x, x_mask=None):
            for c1, c2 in zip(self.convs1, self.convs2):
                xt = F.leaky_relu(x, LRELU_SLOPE)
                xt = c1(xt)
                xt = F.leaky_relu(xt, LRELU_SLOPE)
                xt = c2(xt)
                x = xt + x
            if x_mask is not None:
                x = x * x_mask
            return x

        def remove_weight_norm(self):
            for layer in self.convs1:
                remove_weight_norm(layer)
            for layer in self.convs2:
                remove_weight_norm(layer)

    class ResBlock2(nn.Module):
        def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
            super().__init__()
            self.convs = nn.ModuleList([
                weight_norm(Conv1d(channels, channels, kernel_size, 1,
                    dilation=d, padding=commons.get_padding(kernel_size, d)))
                for d in dilation
            ])

        def forward(self, x, x_mask=None):
            for c in self.convs:
                xt = F.leaky_relu(x, LRELU_SLOPE)
                xt = c(xt)
                x = xt + x
            if x_mask is not None:
                x = x * x_mask
            return x

        def remove_weight_norm(self):
            for layer in self.convs:
                remove_weight_norm(layer)

    class WN(nn.Module):
        def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers,
                     gin_channels=0, p_dropout=0):
            super().__init__()
            self.hidden_channels = hidden_channels
            self.kernel_size = kernel_size
            self.dilation_rate = dilation_rate
            self.n_layers = n_layers
            self.gin_channels = gin_channels
            self.p_dropout = p_dropout

            self.in_layers = nn.ModuleList()
            self.res_skip_layers = nn.ModuleList()
            self.drop = nn.Dropout(p_dropout)

            if gin_channels != 0:
                cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
                self.cond_layer = weight_norm(cond_layer, name='weight')

            for i in range(n_layers):
                dilation = dilation_rate ** i
                padding = int((kernel_size * dilation - dilation) / 2)
                in_layer = nn.Conv1d(hidden_channels, 2 * hidden_channels,
                                     kernel_size, dilation=dilation, padding=padding)
                in_layer = weight_norm(in_layer, name='weight')
                self.in_layers.append(in_layer)
                if i < n_layers - 1:
                    res_skip_channels = 2 * hidden_channels
                else:
                    res_skip_channels = hidden_channels
                res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
                res_skip_layer = weight_norm(res_skip_layer, name='weight')
                self.res_skip_layers.append(res_skip_layer)

        def forward(self, x, x_mask, g=None, **kwargs):
            output = torch.zeros_like(x)
            n_channels_tensor = torch.IntTensor([self.hidden_channels])
            if g is not None:
                g = self.cond_layer(g)
            for i in range(self.n_layers):
                x_in = self.in_layers[i](x)
                if g is not None:
                    cond_offset = i * 2 * self.hidden_channels
                    g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
                else:
                    g_l = torch.zeros_like(x_in)
                acts = self._fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
                acts = self.drop(acts)
                res_skip_acts = self.res_skip_layers[i](acts)
                if i < self.n_layers - 1:
                    res_acts = res_skip_acts[:, :self.hidden_channels, :]
                    x = (x + res_acts) * x_mask
                    output = output + res_skip_acts[:, self.hidden_channels:, :]
                else:
                    output = output + res_skip_acts
            return output * x_mask

        @staticmethod
        def _fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
            n_channels_int = n_channels[0]
            in_act = input_a + input_b
            t_act = torch.tanh(in_act[:, :n_channels_int, :])
            s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
            acts = t_act * s_act
            return acts

        def remove_weight_norm(self):
            if self.gin_channels != 0:
                remove_weight_norm(self.cond_layer)
            for layer in self.in_layers:
                remove_weight_norm(layer)
            for layer in self.res_skip_layers:
                remove_weight_norm(layer)

    class Log(nn.Module):
        def forward(self, x, x_mask, reverse=False, **kwargs):
            if not reverse:
                y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
                logdet = torch.sum(-y, [1, 2])
                return y, logdet
            else:
                x = torch.exp(x) * x_mask
                return x

    class Flip(nn.Module):
        def forward(self, x, *args, reverse=False, **kwargs):
            x = torch.flip(x, [1])
            if not reverse:
                logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
                return x, logdet
            else:
                return x

    class ResidualCouplingLayer(nn.Module):
        def __init__(self, channels, hidden_channels, kernel_size, dilation_rate,
                     n_layers, p_dropout=0, gin_channels=0, mean_only=False):
            super().__init__()
            self.channels = channels
            self.hidden_channels = hidden_channels
            self.kernel_size = kernel_size
            self.dilation_rate = dilation_rate
            self.n_layers = n_layers
            self.half_channels = channels // 2
            self.mean_only = mean_only

            self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
            self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers,
                          p_dropout=p_dropout, gin_channels=gin_channels)
            self.post = nn.Conv1d(hidden_channels,
                                  self.half_channels if mean_only else channels, 1)
            self.post.weight.data.zero_()
            self.post.bias.data.zero_()

        def forward(self, x, x_mask, g=None, reverse=False):
            x0, x1 = torch.split(x, [self.half_channels] * 2, dim=1)
            h = self.pre(x0) * x_mask
            h = self.enc(h, x_mask, g=g)
            stats = self.post(h) * x_mask
            if not self.mean_only:
                m, logs = torch.split(stats, [self.half_channels] * 2, dim=1)
            else:
                m = stats
                logs = torch.zeros_like(m)
            if not reverse:
                x1 = m + x1 * torch.exp(logs) * x_mask
                x = torch.cat([x0, x1], dim=1)
                logdet = torch.sum(logs, [1, 2])
                return x, logdet
            else:
                x1 = (x1 - m) * torch.exp(-logs) * x_mask
                x = torch.cat([x0, x1], dim=1)
                return x

    # Store in namespace
    class _Modules:
        pass

    mods = _Modules()
    mods.ResBlock1 = ResBlock1
    mods.ResBlock2 = ResBlock2
    mods.WN = WN
    mods.Flip = Flip
    mods.ResidualCouplingLayer = ResidualCouplingLayer
    mods.LRELU_SLOPE = LRELU_SLOPE
    return commons, mods


def _build_synthesizer_trn(commons_ns, modules_ns):
    """Reconstruct SynthesizerTrn matching freevc.json config."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import Conv1d, ConvTranspose1d
    from torch.nn.utils import weight_norm, remove_weight_norm

    cm = commons_ns
    md = modules_ns

    class Encoder(nn.Module):
        def __init__(self, in_channels, out_channels, hidden_channels,
                     kernel_size, dilation_rate, n_layers, gin_channels=0):
            super().__init__()
            self.out_channels = out_channels
            self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
            self.enc = md.WN(hidden_channels, kernel_size, dilation_rate,
                              n_layers, gin_channels=gin_channels)
            self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

        def forward(self, x, x_lengths, g=None):
            x_mask = torch.unsqueeze(
                cm.sequence_mask(x_lengths, x.size(2)), 1
            ).to(x.dtype)
            x = self.pre(x) * x_mask
            x = self.enc(x, x_mask, g=g)
            stats = self.proj(x) * x_mask
            m, logs = torch.split(stats, self.out_channels, dim=1)
            z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
            return z, m, logs, x_mask

    class ResidualCouplingBlock(nn.Module):
        def __init__(self, channels, hidden_channels, kernel_size,
                     dilation_rate, n_layers, n_flows=4, gin_channels=0):
            super().__init__()
            self.flows = nn.ModuleList()
            for i in range(n_flows):
                self.flows.append(md.ResidualCouplingLayer(
                    channels, hidden_channels, kernel_size, dilation_rate,
                    n_layers, gin_channels=gin_channels, mean_only=True
                ))
                self.flows.append(md.Flip())

        def forward(self, x, x_mask, g=None, reverse=False):
            if not reverse:
                for flow in self.flows:
                    x, _ = flow(x, x_mask, g=g, reverse=reverse)
            else:
                for flow in reversed(self.flows):
                    x = flow(x, x_mask, g=g, reverse=reverse)
            return x

    class Generator(nn.Module):
        def __init__(self, initial_channel, resblock_type, resblock_kernel_sizes,
                     resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
                     upsample_kernel_sizes, gin_channels=0):
            super().__init__()
            self.num_kernels = len(resblock_kernel_sizes)
            self.num_upsamples = len(upsample_rates)
            self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
            resblock_cls = md.ResBlock1 if resblock_type == '1' else md.ResBlock2

            self.ups = nn.ModuleList()
            for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
                self.ups.append(weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k, u, padding=(k - u) // 2,
                    )
                ))

            self.resblocks = nn.ModuleList()
            for i in range(len(self.ups)):
                ch = upsample_initial_channel // (2 ** (i + 1))
                for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                    self.resblocks.append(resblock_cls(ch, k, d))

            self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
            for up in self.ups:
                up.apply(cm.init_weights)

            if gin_channels != 0:
                self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        def forward(self, x, g=None):
            x = self.conv_pre(x)
            if g is not None:
                x = x + self.cond(g)
            for i in range(self.num_upsamples):
                x = F.leaky_relu(x, md.LRELU_SLOPE)
                x = self.ups[i](x)
                xs = None
                for j in range(self.num_kernels):
                    if xs is None:
                        xs = self.resblocks[i * self.num_kernels + j](x)
                    else:
                        xs += self.resblocks[i * self.num_kernels + j](x)
                x = xs / self.num_kernels
            x = F.leaky_relu(x)
            x = self.conv_post(x)
            return torch.tanh(x)

        def remove_weight_norm(self):
            for layer in self.ups:
                remove_weight_norm(layer)
            for layer in self.resblocks:
                layer.remove_weight_norm()

    class SynthesizerTrn(nn.Module):
        def __init__(self):
            super().__init__()
            # Config from freevc.json + spec_channels = filter_length//2+1 = 641
            inter_channels = 192
            hidden_channels = 192
            gin_channels = 256
            ssl_dim = 1024

            self.enc_p = Encoder(ssl_dim, inter_channels, hidden_channels, 5, 1, 16)
            self.dec = Generator(
                inter_channels, '1',
                [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 8, 2, 2], 512, [16, 16, 4, 4],
                gin_channels=gin_channels,
            )
            self.enc_q = Encoder(_SPEC_CHANNELS, inter_channels, hidden_channels, 5, 1, 16,
                                  gin_channels=gin_channels)
            self.flow = ResidualCouplingBlock(
                inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
            )

        def infer(self, c, g):
            """
            Parameters
            ----------
            c : (1, 1024, T) — WavLM content features (transposed)
            g : (1, 256)     — speaker d-vector

            Returns
            -------
            (1, 1, samples) — waveform

            Notes
            -----
            The prior encoder (enc_p) normally samples z_p = m_p + eps*exp(logs_p).
            We use m_p (the mean) directly for a deterministic and ONNX-exportable
            forward pass.  This matches the typical VITS inference practice.
            """
            c_lengths = torch.tensor([c.size(2)], dtype=torch.long, device=c.device)
            g_3d = g.unsqueeze(-1)   # (1, 256, 1)

            # Use enc_p mean (m_p) directly — deterministic ONNX-friendly path
            _z_p, m_p, _logs_p, c_mask = self.enc_p(c, c_lengths)
            z = self.flow(m_p, c_mask, g=g_3d, reverse=True)
            o = self.dec(z * c_mask, g=g_3d)
            return o

    return SynthesizerTrn


def _build_speaker_encoder():
    """Reconstruct the GE2E speaker encoder (LSTM + Linear).

    Architecture from speaker_encoder/voice_encoder.py:
      mel_n_channels=40, model_hidden_size=256, model_num_layers=3,
      model_embedding_size=256 (= gin_channels in FreeVC)
    """
    import torch
    import torch.nn as nn

    class SpeakerEncoder(nn.Module):
        def __init__(self, mel_n_channels=40, model_hidden_size=256,
                     model_num_layers=3, model_embedding_size=256):
            super().__init__()
            self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
            self.linear = nn.Linear(model_hidden_size, model_embedding_size)
            self.relu = nn.ReLU()

        def forward(self, mels: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            mels : (batch, n_frames, 40) — log-mel spectrogram

            Returns
            -------
            (batch, 256) — L2-normalised speaker embedding
            """
            _, (hidden, _) = self.lstm(mels)
            embeds_raw = self.relu(self.linear(hidden[-1]))
            embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-8)
            return embeds

    return SpeakerEncoder


def _build_wavlm_full():
    """Return a torch module that outputs WavLM-Large FULL final hidden states.

    FreeVC calls: cmodel.extract_features(y)[0]
    which returns res["x"] — the transformer's final output — shape (batch, T, 1024).
    This is different from kNN-VC which extracts layer-6 via hidden_states[7].
    """
    import torch
    import torch.nn as nn
    from transformers import WavLMModel

    class WavLMFull(nn.Module):
        """WavLM-Large returning the final transformer output (full model)."""

        def __init__(self, wavlm: WavLMModel):
            super().__init__()
            self.wavlm = wavlm

        def forward(self, input_values: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            input_values:
                (batch, time) float32 — 16 kHz PCM, normalised.

            Returns
            -------
            torch.Tensor
                (batch, frames, 1024) — final transformer hidden states.
            """
            out = self.wavlm(input_values=input_values)
            return out.last_hidden_state

    print(f"[export] Loading {WAVLM_HF_MODEL} ...")
    wavlm = WavLMModel.from_pretrained(WAVLM_HF_MODEL)
    wavlm.eval()
    model = WavLMFull(wavlm)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"[export] Already cached: {dest}")
        return dest
    print(f"[export] Downloading {url} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest))
    print(f"[export] Saved {dest} ({dest.stat().st_size // 1024 // 1024} MB)")
    return dest


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------


def export_freevc(output_dir: str, cache_dir: Optional[str] = None) -> Path:
    """Export FreeVC components and return the engine output directory."""
    import torch
    from conversion.export_base import OutputLayout, export_model, write_manifest, write_provenance
    from conversion.parity import compare_outputs, check_tolerance, run_ort
    from conversion.quantize import quantize_model

    output_dir = Path(output_dir)
    layout = OutputLayout.for_engine("freevc", base_dir=output_dir)
    layout.makedirs()

    _cache = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "voiceclonnx" / "freevc"
    _cache.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Export WavLM-Large (full final output)
    # ------------------------------------------------------------------
    wavlm_model = _build_wavlm_full()

    dummy_audio = torch.zeros(1, 16000)  # 1 second at 16 kHz

    wavlm_onnx = layout.component_path("wavlm_freevc.onnx")
    print(f"[export] Exporting WavLM-Large (full output) → {wavlm_onnx} ...")
    export_model(
        model=wavlm_model,
        dummy_inputs=(dummy_audio,),
        output_path=wavlm_onnx,
        input_names=["input_values"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_values": {0: "batch", 1: "time"},
            "last_hidden_state": {0: "batch", 1: "frames"},
        },
        opset_version=14,
    )
    print(f"[export] WavLM ONNX written: {wavlm_onnx.stat().st_size / 1024**2:.1f} MB")

    # Parity — WavLM
    print("[export] Running WavLM parity check ...")
    with torch.no_grad():
        torch_wavlm_out = wavlm_model(dummy_audio).detach().cpu().numpy()
    ort_wavlm_out = run_ort(wavlm_onnx, {"input_values": dummy_audio.numpy()})
    wavlm_report = compare_outputs(
        [torch_wavlm_out],
        ort_wavlm_out,
        names=["last_hidden_state"],
        max_abs_tol=1e-3,
        mean_abs_tol=1e-4,
    )
    print("[export] WavLM parity:", wavlm_report.summary())
    check_tolerance(wavlm_report)
    wavlm_report.save(layout.component_path("wavlm_parity_report.json"))

    # Quantize WavLM
    print("[export] Quantizing WavLM (INT8) ...")
    wavlm_q8_path = layout.component_path("wavlm_freevc_q8.onnx")
    wavlm_quant = quantize_model(wavlm_onnx, output_path=wavlm_q8_path)
    print(wavlm_quant.summary())

    del wavlm_model

    # ------------------------------------------------------------------
    # 2. Export speaker encoder (GE2E LSTM)
    # ------------------------------------------------------------------
    spk_ckpt = _download(SPEAKER_ENCODER_URL, _cache / "pretrained_bak_5805000.pt")

    SpeakerEncoder = _build_speaker_encoder()
    smodel = SpeakerEncoder()

    spk_state = torch.load(str(spk_ckpt), map_location="cpu", weights_only=False)
    # The checkpoint may be a full dict or a state dict
    if isinstance(spk_state, dict) and "model_state" in spk_state:
        spk_state = spk_state["model_state"]
    elif isinstance(spk_state, dict) and "lstm.weight_ih_l0" not in spk_state:
        # Try nested keys
        for key in ("state_dict", "model"):
            if key in spk_state:
                spk_state = spk_state[key]
                break
    # pretrained_bak_5805000.pt has similarity_weight/bias keys that belong
    # to the training-time cosine-similarity head — not used at inference.
    smodel.load_state_dict(spk_state, strict=False)
    smodel.eval()

    # dummy: 1 utterance, 160 frames (2.5s at 10ms hop), 40 mel channels
    dummy_mels = torch.zeros(1, 160, 40)
    spk_onnx = layout.component_path("speaker_encoder.onnx")
    print(f"[export] Exporting speaker encoder → {spk_onnx} ...")
    export_model(
        model=smodel,
        dummy_inputs=(dummy_mels,),
        output_path=spk_onnx,
        input_names=["mels"],
        output_names=["embedding"],
        dynamic_axes={
            "mels": {0: "batch", 1: "n_frames"},
            "embedding": {0: "batch"},
        },
        opset_version=14,
    )
    print(f"[export] Speaker encoder ONNX: {spk_onnx.stat().st_size / 1024**2:.1f} MB")

    # Parity — speaker encoder
    print("[export] Running speaker encoder parity check ...")
    with torch.no_grad():
        torch_spk_out = smodel(dummy_mels).detach().cpu().numpy()
    ort_spk_out = run_ort(spk_onnx, {"mels": dummy_mels.numpy()})
    spk_report = compare_outputs(
        [torch_spk_out],
        ort_spk_out,
        names=["embedding"],
        max_abs_tol=1e-4,
        mean_abs_tol=1e-5,
    )
    print("[export] Speaker encoder parity:", spk_report.summary())
    check_tolerance(spk_report)
    spk_report.save(layout.component_path("speaker_encoder_parity_report.json"))

    # Quantize speaker encoder
    print("[export] Quantizing speaker encoder (INT8) ...")
    spk_q8_path = layout.component_path("speaker_encoder_q8.onnx")
    spk_quant = quantize_model(spk_onnx, output_path=spk_q8_path)
    print(spk_quant.summary())

    del smodel

    # ------------------------------------------------------------------
    # 3. Export SynthesizerTrn (VITS decoder / infer path)
    # ------------------------------------------------------------------
    freevc_ckpt = _download(FREEVC_CHECKPOINT_URL, _cache / "freevc.pth")

    commons_ns, modules_ns = _build_architecture()
    SynthesizerTrn = _build_synthesizer_trn(commons_ns, modules_ns)
    net_g = SynthesizerTrn()

    ckpt = torch.load(str(freevc_ckpt), map_location="cpu", weights_only=False)
    # Checkpoint may have 'model' key or be a flat state dict
    state_dict = ckpt.get("model", ckpt)
    # Remove weight_norm from generator for clean export
    net_g.load_state_dict(state_dict, strict=False)
    net_g.eval()

    # Remove weight_norm from generator and encoder
    for module in net_g.modules():
        if hasattr(module, "weight_g"):
            try:
                from torch.nn.utils import remove_weight_norm
                remove_weight_norm(module)
            except Exception:
                pass

    # Dummy inputs: c=(1, 1024, 81 frames), g=(1, 256)
    dummy_c = torch.zeros(1, 1024, 81)
    dummy_g = torch.zeros(1, 256)

    vits_onnx = layout.component_path("freevc_decoder.onnx")
    print(f"[export] Exporting SynthesizerTrn → {vits_onnx} ...")

    class _InferWrapper(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, c, g):
            return self.net.infer(c, g)

    wrapper = _InferWrapper(net_g)
    wrapper.eval()

    export_model(
        model=wrapper,
        dummy_inputs=(dummy_c, dummy_g),
        output_path=vits_onnx,
        input_names=["c", "g"],
        output_names=["waveform"],
        dynamic_axes={
            "c": {0: "batch", 2: "frames"},
            "waveform": {0: "batch", 2: "samples"},
        },
        opset_version=14,
    )
    print(f"[export] Decoder ONNX: {vits_onnx.stat().st_size / 1024**2:.1f} MB")

    # Parity — decoder
    print("[export] Running decoder parity check ...")
    with torch.no_grad():
        torch_dec_out = wrapper(dummy_c, dummy_g).detach().cpu().numpy()
    ort_dec_out = run_ort(vits_onnx, {"c": dummy_c.numpy(), "g": dummy_g.numpy()})
    dec_report = compare_outputs(
        [torch_dec_out],
        ort_dec_out,
        names=["waveform"],
        max_abs_tol=1e-3,
        mean_abs_tol=1e-4,
    )
    print("[export] Decoder parity:", dec_report.summary())
    check_tolerance(dec_report)
    dec_report.save(layout.component_path("decoder_parity_report.json"))

    # Quantize decoder
    print("[export] Quantizing decoder (INT8) ...")
    vits_q8_path = layout.component_path("freevc_decoder_q8.onnx")
    dec_quant = quantize_model(vits_onnx, output_path=vits_q8_path)
    print(dec_quant.summary())

    del net_g, wrapper

    # ------------------------------------------------------------------
    # 4. Write manifest and provenance
    # ------------------------------------------------------------------
    write_manifest(
        layout=layout,
        components={
            "wavlm_encoder": "wavlm_freevc.onnx",
            "wavlm_encoder_q8": "wavlm_freevc_q8.onnx",
            "speaker_encoder": "speaker_encoder.onnx",
            "speaker_encoder_q8": "speaker_encoder_q8.onnx",
            "decoder": "freevc_decoder.onnx",
            "decoder_q8": "freevc_decoder_q8.onnx",
        },
        sample_rates={"input": 16000, "output": 16000},
        metadata={
            "opset": 14,
            "wavlm_source": WAVLM_HF_MODEL,
            "wavlm_extraction": "last_hidden_state (full final output)",
            "wavlm_note": (
                "FreeVC uses extract_features()[0] which is the full transformer "
                "final output — NOT layer-6 as in kNN-VC. wavlm_freevc.onnx "
                "CANNOT be shared with TigreGotico/voiceclonnx-knn-vc."
            ),
            "freevc_checkpoint": FREEVC_CHECKPOINT_URL,
            "speaker_encoder_checkpoint": SPEAKER_ENCODER_URL,
            "sampling_rate": str(_SAMPLING_RATE),
        },
    )
    write_provenance(
        engine_dir=layout.engine_dir,
        upstream_repo_url=FREEVC_UPSTREAM_URL,
        upstream_ref=FREEVC_UPSTREAM_REF,
        license_text=MIT_LICENSE,
        extra={
            "wavlm_model": WAVLM_HF_MODEL,
            "wavlm_extraction": "last_hidden_state",
            "freevc_checkpoint_source": FREEVC_CHECKPOINT_URL,
            "speaker_encoder_source": SPEAKER_ENCODER_URL,
        },
    )

    print(f"\n[export] FreeVC export complete → {layout.engine_dir}")
    return layout.engine_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export FreeVC (WavLM-Large + Speaker Encoder + VITS decoder) to ONNX."
    )
    p.add_argument("--output-dir", default="/tmp/freevc-out", help="Staging output directory.")
    p.add_argument("--cache-dir", default=None, help="Cache directory for downloaded checkpoints.")
    p.add_argument("--no-push", action="store_true", help="Skip HF upload.")
    p.add_argument("--dry-run-push", action="store_true", help="Dry-run HF upload.")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    engine_dir = export_freevc(output_dir=args.output_dir, cache_dir=args.cache_dir)

    if not args.no_push:
        from conversion.push_models import push_engine
        push_engine(
            engine_dir=engine_dir,
            engine_name="freevc",
            dry_run=args.dry_run_push,
            commit_message="export: add freevc ONNX artifacts (WavLM-Large full + speaker encoder + VITS decoder)",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
