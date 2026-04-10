from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import soxr
from pesq import BufferTooShortError, NoUtterancesError, pesq


@dataclass(frozen=True)
class AudioPair:
    reference: np.ndarray
    estimate: np.ndarray
    fs: int


def load_mono_audio(path: str, target_fs: int = 16000) -> np.ndarray:
    data, fs = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data[:, :1]
    if fs != target_fs:
        data = soxr.resample(data, fs, target_fs)
    return data.T


def align_lengths(reference: np.ndarray, estimate: np.ndarray) -> AudioPair:
    if reference.ndim != 2 or estimate.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got {reference.shape} and {estimate.shape}.")
    length = min(reference.shape[-1], estimate.shape[-1])
    if length == 0:
        raise ValueError("Audio length is zero after alignment.")
    return AudioPair(reference=reference[..., :length], estimate=estimate[..., :length], fs=16000)


def compute_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    if reference.shape != estimate.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {estimate.shape}.")
    numerator = np.square(reference).sum().clip(min=1e-7)
    denominator = np.square(estimate - reference).sum().clip(min=1e-7)
    return float(10.0 * np.log10(numerator / denominator))


def compute_pesq(reference: np.ndarray, estimate: np.ndarray, fs: int = 16000) -> float:
    scores: list[float] = []
    for ref_ch, est_ch in zip(reference, estimate, strict=True):
        try:
            score = pesq(fs, ref_ch, est_ch, "wb")
        except (BufferTooShortError, NoUtterancesError) as exc:
            raise RuntimeError(f"PESQ failed: {exc}") from exc
        scores.append(float(score))
    return float(sum(scores) / len(scores))


def evaluate(enhanced_path: str, reference_path: str) -> dict[str, float]:
    reference = load_mono_audio(reference_path)
    estimate = load_mono_audio(enhanced_path)
    pair = align_lengths(reference, estimate)
    return {
        "sdr": compute_sdr(pair.reference, pair.estimate),
        "pesq": compute_pesq(pair.reference, pair.estimate, pair.fs),
        "fs": float(pair.fs),
        "samples": float(pair.reference.shape[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one enhanced audio file with PESQ and SDR.")
    parser.add_argument("--enhanced", default=r"F:/ksl/addse/enhanced_SA1.wav", help="Enhanced audio path.")
    parser.add_argument("--reference", default=r"F:/ksl/TIMIT_all_wavs/SA1.WAV", help="Clean reference path.")
    args = parser.parse_args()

    result = evaluate(args.enhanced, args.reference)
    print("Evaluation result:")
    print(f"  enhanced: {args.enhanced}")
    print(f"  reference: {args.reference}")
    print(f"  fs: {int(result['fs'])}")
    print(f"  samples: {int(result['samples'])}")
    print(f"  PESQ: {result['pesq']:.4f}")
    print(f"  SDR: {result['sdr']:.4f}")


if __name__ == "__main__":
    main()