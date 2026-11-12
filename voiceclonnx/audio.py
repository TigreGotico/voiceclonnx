"""Audio file input and output shared by every engine.

Engines read float32 mono at their own rate and write 16-bit PCM WAV.
"""

from __future__ import annotations

import numpy as np


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample *audio* from *orig_sr* to *target_sr* by linear interpolation."""
    if orig_sr == target_sr:
        return audio
    n_out = int(round(len(audio) * target_sr / orig_sr))
    return np.interp(np.linspace(0, len(audio) - 1, n_out), np.arange(len(audio)), audio).astype(np.float32)


def load_wav(path: str, target_sr: int) -> np.ndarray:
    """Load *path* as float32 mono at *target_sr*."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return resample(audio, sr, target_sr).astype(np.float32)


def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """Write float32 *audio* to *path* as 16-bit PCM WAV."""
    import soundfile as sf

    sf.write(str(path), (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16), sr, subtype="PCM_16")
