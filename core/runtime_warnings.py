from __future__ import annotations

import logging
import warnings


def suppress_pyannote_tf32_warning() -> None:
    """Hide noisy third-party runtime warnings without changing model behavior."""
    suppress_noisy_runtime_logs()
    warnings.filterwarnings("ignore", message=r".*TensorFloat-32.*")
    warnings.filterwarnings("ignore", message=r".*TF32.*")
    try:
        from pyannote.audio.utils.reproducibility import ReproducibilityWarning
    except Exception:  # noqa: BLE001
        return

    warnings.filterwarnings("ignore", category=ReproducibilityWarning)


def suppress_noisy_runtime_logs() -> None:
    warnings.filterwarnings("ignore", message=r".*legacy format.*torch\.export\.save.*")
    warnings.filterwarnings("ignore", message=r".*Please generate a new pt2 file.*")
    warnings.filterwarnings("ignore", message=r".*torch\.jit\.load.*deprecated.*")
    warnings.filterwarnings("ignore", message=r".*If you intend to do training or fine-tuning.*")
    warnings.filterwarnings("ignore", message=r".*setup_training_data.*")
    warnings.filterwarnings("ignore", message=r".*ModelPT.*")
    try:
        from nemo.utils import logging as nemo_logging
        nemo_logging.setLevel(logging.ERROR)
        nemo_logging.set_verbosity(logging.ERROR)
    except Exception:
        pass
    for logger_name in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "huggingface_hub.file_download",
        "torch.export",
        "torch.export.pt2_archive",
        "torch.export.pt2_archive._package",
        "nemo_logging",
        "nemo",
        "nemo.collections",
        "nemo.core",
        "nemo.utils",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR if "nemo" in logger_name else logging.WARNING)

