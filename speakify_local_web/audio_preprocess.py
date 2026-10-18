
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch

SR, MELS, FRAMES = 16000, 64, 1536


def _ffmpeg_bin():
    configured = os.getenv("FFMPEG_PATH", "").strip()
    if configured and Path(configured).exists():
        return configured
    found = shutil.which("ffmpeg")
    if found:
        return found
    windows_local = Path(
        r"D:\Downnload\ffmpeg-9.0.2-essentials_build\ffmpeg-9.0.2-essentials_build\bin\ffmpeg.exe"
    )
    if windows_local.exists():
        return str(windows_local)
    raise RuntimeError(
        "Không tìm thấy FFmpeg. Cài FFmpeg rồi thêm vào PATH, "
        "hoặc đặt FFMPEG_PATH trong file .env."
    )


def wav_to_tensor(path):
    wav_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as f:
            wav_path = f.name

        result = subprocess.run(
            [
                _ffmpeg_bin(),
                "-y",
                "-i", str(path),
                "-vn",
                "-ac", "1",
                "-ar", str(SR),
                "-c:a", "pcm_s16le",
                wav_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise ValueError(
                "Không chuyển được âm thanh sang WAV: "
                + result.stderr[-1500:]
            )

        y, _ = librosa.load(
            wav_path, sr=SR, mono=True
        )

        if not y.size:
            raise ValueError("Không tìm thấy âm thanh.")

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=SR,
            n_fft=400,
            hop_length=160,
            win_length=400,
            n_mels=MELS,
            power=2.0,
        )

        feat = librosa.power_to_db(
            mel, ref=np.max
        ).astype(np.float32)

        n = min(feat.shape[1], FRAMES)
        clipped = feat[:, :n]

        mean = float(clipped.mean()) if clipped.size else 0.0
        std = float(clipped.std()) if clipped.size else 1.0

        out = np.zeros((MELS, FRAMES), dtype=np.float32)

        if n:
            out[:, :n] = (
                clipped - mean
            ) / max(std, 1e-6)

        return torch.from_numpy(out).unsqueeze(0)

    finally:
        if wav_path and Path(wav_path).exists():
            Path(wav_path).unlink()
