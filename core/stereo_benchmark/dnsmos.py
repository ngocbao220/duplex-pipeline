"""Local Microsoft DNSMOS P.835 scoring with two checked-in-external ONNX assets."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


from core.compat import ensure_runtime_compat

SAMPLE_RATE = 16000
WINDOW_SEC = 9.01
PRIMARY_MODEL = "sig_bak_ovr.onnx"
P808_MODEL = "model_v8.onnx"
ONNX_BATCH_SIZE = 16


def resolve_dnsmos_dir(model_dir: Path | str | None) -> Path:
    """Find local DNSMOS assets without downloading or using a remote fallback."""
    if not model_dir or str(model_dir).strip() in ("", "."):
        model_dir = Path("models/dnsmos")
    else:
        model_dir = Path(model_dir)

    if (model_dir / PRIMARY_MODEL).is_file() and (model_dir / P808_MODEL).is_file():
        return model_dir.resolve()

    # Check DUPLEX_MODEL_DIR environment variable
    base_dir = os.environ.get("DUPLEX_MODEL_DIR")
    if base_dir:
        cand = Path(base_dir) / "dnsmos"
        if (cand / PRIMARY_MODEL).is_file() and (cand / P808_MODEL).is_file():
            return cand.resolve()
        cand_base = Path(base_dir)
        if (cand_base / PRIMARY_MODEL).is_file() and (cand_base / P808_MODEL).is_file():
            return cand_base.resolve()

    # Check local subdirectories (e.g. when a mounted model dataset nests the files).
    if model_dir.exists():
        found = list(model_dir.rglob(PRIMARY_MODEL))
        for match in found:
            parent = match.parent
            if (parent / PRIMARY_MODEL).is_file() and (parent / P808_MODEL).is_file():
                return parent.resolve()
        # If model_dir exists but lacks assets, do not redirect to global fallback
        return model_dir.resolve()

    fallback_dir = Path("models/dnsmos")
    if (fallback_dir / PRIMARY_MODEL).is_file() and (fallback_dir / P808_MODEL).is_file():
        return fallback_dir.resolve()

    return model_dir.resolve()


class DNSMOSScorer:
    """Reference-free DNSMOS P.835 scorer; higher SIG/BAK/OVRL/P808 values are better.

    This reproduces Microsoft's local P.835 preprocessing: 16 kHz, 9.01-second
    windows on one-second hops, and the official non-personalized calibration.
    """

    def __init__(self, model_dir: Path):
        ensure_runtime_compat()
        self.model_dir = resolve_dnsmos_dir(Path(model_dir))
        self.error: str | None = self._missing_assets()
        self.primary = None
        self.p808 = None
        if self.error is None:
            try:
                import os
                import onnxruntime as ort

                os.environ["ORT_DISABLE_TELEMETRY"] = "1"
                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1

                primary_path = str((self.model_dir / PRIMARY_MODEL).resolve())
                p808_path = str((self.model_dir / P808_MODEL).resolve())
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
                try:
                    self.primary = ort.InferenceSession(primary_path, sess_options=opts, providers=providers)
                    self.p808 = ort.InferenceSession(p808_path, sess_options=opts, providers=providers)
                except Exception:
                    self.primary = ort.InferenceSession(primary_path, sess_options=opts, providers=["CPUExecutionProvider"])
                    self.p808 = ort.InferenceSession(p808_path, sess_options=opts, providers=["CPUExecutionProvider"])
            except Exception as error:  # model/runtime is optional for the complete benchmark
                self.error = f"DNSMOS unavailable: {type(error).__name__}: {error}"

    def score(self, audio: np.ndarray, sample_rate: int) -> dict:
        """Score one mono track without any clean reference signal."""
        if self.error is not None:
            return {"status": "unavailable", "reason": self.error}
        try:
            # Some server launchers export an empty cache location; numba/librosa
            # treats that as a directory creation request for "".
            for cache_var in ("NUMBA_CACHE_DIR", "LIBROSA_CACHE_DIR", "XDG_CACHE_HOME"):
                if os.environ.get(cache_var) == "":
                    os.environ.pop(cache_var)
            import librosa

            signal = np.asarray(audio, dtype=np.float32)
            if sample_rate != SAMPLE_RATE:
                signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
            if not signal.size:
                return {"status": "unavailable", "reason": "DNSMOS input is empty"}
            window_samples = int(WINDOW_SEC * SAMPLE_RATE)
            while signal.size < window_samples:
                signal = np.append(signal, signal)
            
            num_hops = int(np.floor(signal.size / SAMPLE_RATE) - WINDOW_SEC) + 1
            if num_hops <= 0:
                return {"status": "unavailable", "reason": "DNSMOS produced no complete analysis window"}

            # Compute full mel-spectrogram once for the entire signal (1000x faster than per-chunk STFT)
            full_mel = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_fft=321, hop_length=160, n_mels=120)
            
            primary_input_name = self.primary.get_inputs()[0].name
            p808_input_name = self.p808.get_inputs()[0].name
            scores = []
            for batch_start in range(0, num_hops, ONNX_BATCH_SIZE):
                chunks, features = [], []
                for i in range(batch_start, min(batch_start + ONNX_BATCH_SIZE, num_hops)):
                    audio_chunk = signal[i * SAMPLE_RATE : i * SAMPLE_RATE + window_samples]
                    mel_slice = full_mel[:, i * 100 : i * 100 + 900]
                    if audio_chunk.size != window_samples or mel_slice.shape[1] != 900:
                        continue
                    chunks.append(audio_chunk)
                    features.append(((librosa.power_to_db(mel_slice, ref=np.max) + 40) / 40).T)
                if not chunks:
                    continue
                raw_primary = self.primary.run(None, {primary_input_name: np.asarray(chunks, dtype=np.float32)})[0]
                raw_p808 = self.p808.run(None, {p808_input_name: np.asarray(features, dtype=np.float32)})[0]
                for (raw_sig, raw_bak, raw_ovrl), p808_val in zip(raw_primary, np.asarray(raw_p808).reshape(-1)):
                    sig, bak, ovrl = self._calibrate(raw_sig, raw_bak, raw_ovrl)
                    scores.append((sig, bak, ovrl, float(p808_val)))

            if not scores:
                return {"status": "unavailable", "reason": "DNSMOS produced no complete analysis window"}
                
            mean = np.mean(np.asarray(scores), axis=0)
            return {
                "status": "ok", "sig": float(mean[0]), "bak": float(mean[1]), "ovrl": float(mean[2]),
                "p808_mos": float(mean[3]), "window_count": len(scores),
            }
        except Exception as error:
            return {"status": "unavailable", "reason": f"DNSMOS unavailable: {type(error).__name__}: {error}"}

    def _missing_assets(self) -> str | None:
        missing = [name for name in (PRIMARY_MODEL, P808_MODEL) if not (self.model_dir / name).is_file()]
        if missing:
            return f"DNSMOS model asset missing: {', '.join(missing)} in {self.model_dir}"
        return None

    @staticmethod
    def _mel_features(librosa, audio: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_fft=321, hop_length=160, n_mels=120)
        return ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T

    @staticmethod
    def _calibrate(sig: float, bak: float, ovrl: float) -> tuple[float, float, float]:
        # Official DNSMOS P.835 non-personalized calibration polynomials.
        return (
            float(np.poly1d([-0.08397278, 1.22083953, 0.0052439])(sig)),
            float(np.poly1d([-0.13166888, 1.60915514, -0.39604546])(bak)),
            float(np.poly1d([-0.06766283, 1.11546468, 0.04602535])(ovrl)),
        )
