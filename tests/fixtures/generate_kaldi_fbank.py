"""Write the reference CAM++ features that tests/test_cosyvoice.py compares against.

The reference is torchaudio's Kaldi fbank, the call CosyVoice makes for CAM++,
followed by mean normalisation over time. Run it with torch and torchaudio installed:

    python tests/fixtures/generate_kaldi_fbank.py
"""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.compliance.kaldi as kaldi

ROOT = Path(__file__).resolve().parents[2]
wav, sr = sf.read(ROOT / "demo" / "source.wav", dtype="float32")
clip = wav[16000:48000]
feats = kaldi.fbank(torch.from_numpy(clip).unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=sr)
feats = feats - feats.mean(dim=0, keepdim=True)
np.save(Path(__file__).parent / "kaldi_fbank_source_1s_to_3s.npy", feats.numpy().astype(np.float32))
