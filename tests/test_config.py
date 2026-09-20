from __future__ import annotations

import tomllib
import sys
from pathlib import Path

from core.config import load_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "duplexchat" / "src"))
from duplexchat.model_options import infer_diarization_backend, resolve_model_alias, DIARIZATION_MODELS


def test_shared_config_contains_only_integrated_runtime_settings(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        """{
          "runtime": {"device": "cpu", "allow_cpu_fallback": true},
          "separation": {"num_steps": 8},
          "benchmark": {"output_dir": "reports"},
          "cholimex": {"overlap_padding": 0.2, "vad_padding_ms": 80}
        }"""
    )

    config = load_config(path)

    assert config.runtime_device == "cpu"
    assert config.separation_num_steps == 8
    assert config.cholimex_overlap_padding == 0.2
    assert config.cholimex_vad_padding_ms == 80
    assert config.cholimex_speaker_assignment_mode == "relative_similarity"


def test_streaming_sortformer_v21_model_option_resolves_to_sortformer():
    model = resolve_model_alias("sortformer-streaming-v2.1", DIARIZATION_MODELS)

    assert model == "nvidia/diar_streaming_sortformer_4spk-v2.1"
    assert infer_diarization_backend(model) == "sortformer"


def test_shared_config_rejects_removed_crawler_settings(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"source": {"youtube_only": true}}')

    try:
        load_config(path)
    except ValueError as error:
        assert "unknown config key" in str(error)
    else:
        raise AssertionError("crawler configuration must not be accepted")


def test_sommelier_runtime_declares_vendor_import_dependencies():
    """The original entry point imports these before processing CLI flags."""
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pipeline" / "sommelier" / "pyproject.toml").read_text())
    dependencies = set(config["project"]["dependencies"])
    names = {dependency.split("[", 1)[0].split("=", 1)[0] for dependency in dependencies}
    assert {"openai", "onnxruntime", "faster-whisper", "whisperx"} <= names


def test_sommelier_runtime_pins_upstream_torch_compatibility_set():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pipeline" / "sommelier" / "pyproject.toml").read_text())
    dependencies = set(config["project"]["dependencies"])
    assert {"torch==2.7.1", "torchaudio==2.7.1", "torchmetrics==1.7.4"} <= dependencies


def test_sommelier_runtime_pins_hub_version_that_accepts_use_auth_token():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pipeline" / "sommelier" / "pyproject.toml").read_text())
    assert "huggingface-hub==0.33.4" in set(config["project"]["dependencies"])


def test_sommelier_runtime_pins_upstream_speechbrain_and_adapts_legacy_keyword():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pipeline" / "sommelier" / "pyproject.toml").read_text())
    assert "speechbrain==1.0.3" in set(config["project"]["dependencies"])
    source = (root / "pipeline" / "sommelier" / "vendor" / "podcast_pipeline" / "main_original_ASR_MoE.py").read_text()
    assert 'kwargs.pop("use_auth_token", None)' in source


def test_sommelier_defers_salm_import_when_asr_moe_is_disabled():
    root = Path(__file__).resolve().parents[1]
    source = (root / "pipeline" / "sommelier" / "vendor" / "podcast_pipeline" / "main_original_ASR_MoE.py").read_text()
    import_statement = "from nemo.collections.speechlm2.models import SALM"
    assert source.count(import_statement) == 1
    assert source.index(import_statement) > source.index("if args.ASRMoE:")


def test_sommelier_lightning_load_compatibility_wrapper_accepts_weights_only():
    root = Path(__file__).resolve().parents[1]
    source = (root / "pipeline" / "sommelier" / "vendor" / "podcast_pipeline" / "main_original_ASR_MoE.py").read_text()
    assert "def _patched_load(path_or_url: Union[IO, str, Path], map_location=None, weights_only=None)" in source
    assert "torch.load(path_or_url, map_location=map_location, weights_only=False)" in source


def test_sommelier_runner_stops_after_the_two_track_stage():
    root = Path(__file__).resolve().parents[1]
    runner = (root / "pipeline" / "sommelier" / "src" / "sommelier" / "runner.py").read_text()
    vendor = (root / "pipeline" / "sommelier" / "vendor" / "podcast_pipeline" / "main_original_ASR_MoE.py").read_text()
    assert '"--until-pre-asr"' in runner
    assert '"--expected-speakers", "2"' in runner
    assert "if args.until_pre_asr:" in vendor
    assert "return export_pre_asr_result(" in vendor
    assert "def constrain_speaker_inventory(" in vendor


def test_shared_config_loads_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
runtime:
  device: cpu
  allow_cpu_fallback: true
separation:
  num_steps: 12
cholimex:
  overlap_padding: 0.15
"""
    )
    config = load_config(path)
    assert config.runtime_device == "cpu"
    assert config.separation_num_steps == 12
    assert config.cholimex_overlap_padding == 0.15


def test_dialogue_split_is_decoupled_from_duplexchat():
    root = Path(__file__).resolve().parents[1]
    import yaml
    
    with open(root / "configs" / "pipeline" / "duplexchat.yaml", encoding="utf-8") as f:
        duplex_cfg = yaml.safe_load(f)
    assert "music_filter" not in duplex_cfg
    assert "lid" not in duplex_cfg
    assert "separation" in duplex_cfg
    assert duplex_cfg["separation"]["backend"] == "dialoguesidon"

    with open(root / "configs" / "dialogue_split" / "default.yaml", encoding="utf-8") as f:
        split_cfg = yaml.safe_load(f)
    assert "music_filter" in split_cfg
    assert "lid" in split_cfg
    assert "dialogue" in split_cfg


def test_kaggle_dev_config_paths():
    root = Path(__file__).resolve().parents[1]
    import yaml
    
    with open(root / "configs" / "env" / "dev.yaml", encoding="utf-8") as f:
        dev_cfg = yaml.safe_load(f)
    paths = dev_cfg["paths"]
    assert paths["sepreformer"] == "/kaggle/input/models/ngocbaotrinhtuan/sepreformer-base-wsj0/pytorch/default/1/sepreformer_base_wsj0.pth"
    assert paths["dnsmos"] == "/kaggle/input/models/ngocbaotrinhtuan/dnsmos/pytorch/dnsmos/1"
    assert paths["youtube"] == "/kaggle/input/datasets/ngocbaotrinhtuan/youtube"
    assert paths["podcast_index"] == "/kaggle/input/datasets/ngocbaotrinhtuan/podcast-index"
    assert paths["base_output"] == "outputs/processed"
    assert dev_cfg["offline"] is False


def test_sommelier_pipeline_declares_max_chunk_duration():
    root = Path(__file__).resolve().parents[1]
    import yaml
    
    with open(root / "configs" / "pipeline" / "sommelier.yaml", encoding="utf-8") as f:
        somm_cfg = yaml.safe_load(f)
    assert somm_cfg.get("max_chunk_duration") == 300.0


def test_setup_env_has_no_external_index_url():
    root = Path(__file__).resolve().parents[1]
    setup_script = (root / "setup_env.sh").read_text()
    assert "--extra-index-url" not in setup_script
    assert "--index-url" not in setup_script


def test_all_runtime_dependencies_are_pinned():
    root = Path(__file__).resolve().parents[1]
    
    # duplexchat pyproject.toml
    duplex_cfg = tomllib.loads((root / "pipeline" / "duplexchat" / "pyproject.toml").read_text())
    for dep in duplex_cfg["project"]["dependencies"]:
        if dep == "duplex-pipelines":
            continue
        assert "==" in dep, f"Dependency '{dep}' in duplexchat is not strictly pinned with '=='"

    # sommelier pyproject.toml
    somm_cfg = tomllib.loads((root / "pipeline" / "sommelier" / "pyproject.toml").read_text())
    for dep in somm_cfg["project"]["dependencies"]:
        if dep == "duplex-pipelines":
            continue
        assert "==" in dep, f"Dependency '{dep}' in sommelier is not strictly pinned with '=='"

    # requirements-duplexchat.txt
    for line in (root / "requirements-duplexchat.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert "==" in line, f"Dependency '{line}' in requirements-duplexchat.txt is not strictly pinned with '=='"

    # requirements-sommelier.txt
    for line in (root / "requirements-sommelier.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert "==" in line, f"Dependency '{line}' in requirements-sommelier.txt is not strictly pinned with '=='"
