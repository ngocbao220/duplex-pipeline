import numpy as np
import pytest
import soundfile as sf

from core.stereo_benchmark.activity import activity_summary
from core.stereo_benchmark.audio import load_stereo
from core.stereo_benchmark.dynamics import analyze_turns, cosine_distinctiveness
from core.stereo_benchmark.models import acoustic_metrics
from core.stereo_benchmark.dnsmos import DNSMOSScorer
from core.stereo_benchmark.report import render_tables, summarize_reports
from core.stereo_benchmark.runner import discover_corpus_audio


def test_load_stereo_rejects_mono(tmp_path):
    path = tmp_path / "mono.wav"
    sf.write(path, np.zeros(160), 16000)

    with pytest.raises(ValueError, match="exactly two channels"):
        load_stereo(path)


def test_load_stereo_preserves_left_and_right(tmp_path):
    path = tmp_path / "stereo.wav"
    data = np.column_stack([np.full(160, 0.25), np.full(160, -0.5)])
    sf.write(path, data, 16000, subtype="FLOAT")

    audio = load_stereo(path)

    assert audio.sample_rate == 16000
    assert audio.left[0] == pytest.approx(0.25)
    assert audio.right[0] == pytest.approx(-0.5)


def test_activity_summary_partitions_all_frames():
    left = np.array([1, 1, 0, 0], dtype=bool)
    right = np.array([0, 1, 1, 0], dtype=bool)

    result = activity_summary(left, right, frame_sec=0.1)

    assert result["left_only"]["duration_sec"] == pytest.approx(0.1)
    assert result["right_only"]["duration_sec"] == pytest.approx(0.1)
    assert result["overlap"]["duration_sec"] == pytest.approx(0.1)
    assert result["silence"]["duration_sec"] == pytest.approx(0.1)


def test_itd_is_one_minus_cosine_similarity():
    assert cosine_distinctiveness(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
    assert cosine_distinctiveness(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(0.0)


def test_turns_detect_transition_and_overlap_transition():
    left = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
    right = np.array([0, 0, 1, 1, 1, 1], dtype=bool)

    result = analyze_turns(left, right, frame_sec=0.5)

    assert result["transition_count"] == 1
    assert result["overlapping_transition_count"] == 1


def test_short_spurt_inside_other_turn_is_backchannel_candidate():
    left = np.array([1] * 10, dtype=bool)
    right = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=bool)

    result = analyze_turns(left, right, frame_sec=0.2)

    assert len(result["backchannel_candidates"]) == 1
    assert result["backchannel_candidates"][0]["speaker"] == "right"


def test_optional_model_failure_is_reported_without_crashing(monkeypatch):
    import core.stereo_benchmark.models as models

    monkeypatch.setattr(models, "_squim_model", lambda _device: (_ for _ in ()).throw(RuntimeError("model unavailable")))

    result = acoustic_metrics(np.zeros(160), np.zeros(160), "cpu")

    assert result["squim"]["left"]["status"] == "unavailable"
    assert "model unavailable" in result["squim"]["left"]["reason"]


def test_dnsmos_scorer_is_reused_for_multiple_metric_calls(monkeypatch, tmp_path):
    import core.stereo_benchmark.models as models

    created = []

    class FakeScorer:
        def __init__(self, model_dir):
            created.append(model_dir)

        def score(self, _audio, _sample_rate):
            return {"status": "ok", "ovrl": 3.0}

    models._dnsmos_scorer.cache_clear()
    monkeypatch.setattr("core.stereo_benchmark.dnsmos.DNSMOSScorer", FakeScorer)

    acoustic_metrics(np.ones(16000), np.ones(16000), "cpu", tmp_path)
    acoustic_metrics(np.ones(16000), np.ones(16000), "cpu", tmp_path)

    assert created == [tmp_path]


def test_squim_batches_equal_length_windows(monkeypatch):
    import torch
    import core.stereo_benchmark.models as models

    batches = []

    class FakeModel:
        def __call__(self, audio):
            batches.append(audio.shape[0])
            return torch.ones(audio.shape[0]), torch.full((audio.shape[0],), 2.0), torch.full((audio.shape[0],), 3.0)

    monkeypatch.setattr(models, "_squim_model", lambda _device: FakeModel())

    result = acoustic_metrics(np.ones(16000 * 20), np.ones(16000 * 20), "cpu")

    assert batches == [4]
    assert result["squim"]["mean"] == {"sq_stoi": 1.0, "sq_pesq": 2.0, "sq_si_sdr": 3.0}


def test_speaker_encoder_batches_windows_from_both_channels(monkeypatch):
    import torch
    import core.stereo_benchmark.models as models

    batches = []

    class FakeEncoder:
        def encode_batch(self, audio):
            batches.append(audio.shape[0])
            return torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 1.0]]])

    monkeypatch.setattr(models, "_speaker_encoder", lambda _device: FakeEncoder())

    result = models.speaker_metrics(np.ones(16000 * 6), np.ones(16000 * 6), "cpu")

    assert batches == [4]
    assert result["itc"]["mean"]["status"] == "ok"
    assert result["itd"]["status"] == "ok"


def test_benchmark_reuses_prepared_speech_for_acoustic_and_speaker_metrics(monkeypatch, tmp_path):
    import core.stereo_benchmark.runner as runner

    audio_path = tmp_path / "stereo.wav"
    sf.write(audio_path, np.ones((16000, 2)), 16000)
    prepared, calls = [], []

    def fake_prepare(audio, *_args):
        value = np.asarray(audio, dtype=np.float32)
        prepared.append(value)
        return value

    def fake_acoustic(left, right, *_args):
        calls.append((left, right))
        return {"dnsmos": {"left": {"status": "unavailable"}}, "squim": {"left": {"status": "unavailable"}}}

    def fake_speaker(left, right, *_args):
        calls.append((left, right))
        return {"itc": {"mean": {"status": "unavailable"}}, "itd": {"status": "unavailable"}}

    monkeypatch.setattr(runner, "prepare_speech", fake_prepare)
    monkeypatch.setattr(runner, "acoustic_metrics", fake_acoustic)
    monkeypatch.setattr(runner, "speaker_metrics", fake_speaker)

    report, _ = runner.run_benchmark(audio_path, tmp_path / "report", device="cpu")

    assert len(prepared) == 2
    assert calls[0][0] is calls[1][0]
    assert calls[0][1] is calls[1][1]
    assert report["runtime"]["total_seconds"] >= 0


def test_dnsmos_reports_missing_model_assets_without_crashing(tmp_path):
    result = DNSMOSScorer(tmp_path).score(np.zeros(16000), 16000)

    assert result["status"] == "unavailable"
    assert "sig_bak_ovr.onnx" in result["reason"]


def test_corpus_discovery_recurses_over_supported_audio_files(tmp_path):
    _stereo = tmp_path / "nested" / "audio.stereo.wav"
    _stereo.parent.mkdir()
    sf.write(_stereo, np.zeros((160, 2)), 16000)
    (tmp_path / "notes.txt").write_text("not audio")

    assert discover_corpus_audio(tmp_path) == [_stereo]


def test_summary_and_markdown_tables_include_requested_metrics():
    report = {
        "input": {"path": "/tmp/sample.wav", "duration_sec": 60.0},
        "acoustic_quality": {"dnsmos": {"left": {"status": "ok", "ovrl": 3.0}, "right": {"status": "ok", "ovrl": 4.0}, "mean": {"ovrl": 3.5}}, "squim": {"left": {"status": "ok", "sq_stoi": 0.9, "sq_pesq": 3.1}, "right": {"status": "ok", "sq_stoi": 0.8, "sq_pesq": 2.9}, "mean": {"sq_stoi": 0.85, "sq_pesq": 3.0}}},
        "speaker_identity": {"itc": {"left": {"status": "ok", "itc": 0.7}, "right": {"status": "ok", "itc": 0.8}, "mean": {"status": "ok", "itc": 0.75}}, "itd": {"status": "ok", "itd": 0.2}},
        "speech_activity": {"overlap": {"percentage": 10.0}},
        "turn_taking": {"turn_exchanges_per_min": 5.0, "mean_turn_duration_sec": 2.0, "mean_left_turn_duration_sec": 2.1, "mean_right_turn_duration_sec": 1.9, "overlapping_transition_rate": 0.25, "backchannels_per_min": 1.0},
    }

    summary = summarize_reports([report])
    rendered = render_tables(summary)

    assert summary["sample_count"] == 1
    assert "DNSMOS" in rendered
    assert "SQ-STOI" in rendered
    assert "Turn Exchange" in rendered
    assert "Overlap Transition" in rendered
    assert "Meaning" in rendered


def test_check_benchmark_models_preflight(tmp_path):
    from core.stereo_benchmark.preflight import check_benchmark_models, print_model_check_summary

    results = check_benchmark_models(tmp_path)

    assert "dnsmos" in results
    assert "speechbrain_ecapa" in results
    assert "squim" in results
    assert "nisqa" in results
    print_model_check_summary(results)


def test_strict_preflight_requires_every_metric_to_pass():
    from core.stereo_benchmark.preflight import benchmark_models_ready

    assert benchmark_models_ready({
        "dnsmos": {"status": "OK"},
        "speechbrain_ecapa": {"status": "OK"},
        "squim": {"status": "OK"},
        "nisqa": {"status": "OK"},
    })
    assert not benchmark_models_ready({
        "dnsmos": {"status": "OK"},
        "speechbrain_ecapa": {"status": "MISSING"},
        "squim": {"status": "OK"},
        "nisqa": {"status": "OK"},
    })


def test_nisqa_requires_explicit_local_checkpoint(monkeypatch):
    import core.stereo_benchmark.models as models

    monkeypatch.setattr(models, "_ensure_nisqa_installed", lambda: True)
    monkeypatch.setattr(models, "_load_nisqa_model_class", lambda: object)
    monkeypatch.delenv("NISQA_MODEL_PATH", raising=False)
    monkeypatch.delenv("DUPLEX_MODEL_DIR", raising=False)

    result = models._nisqa_one(np.ones(16000, dtype=np.float32), "cpu")

    assert result["status"] == "unavailable"
    assert "NISQA_MODEL_PATH" in result["reason"]
    assert "pretrained_model" not in result["reason"]


def test_nisqa_prediction_args_supply_missing_checkpoint_defaults(tmp_path):
    import core.stereo_benchmark.models as models

    args = models._nisqa_prediction_args(tmp_path / "nisqa.tar", tmp_path / "input.wav")

    assert args["pretrained_model"] == str(tmp_path / "nisqa.tar")
    assert args["ms_channel"] is None
    assert args["tr_bs_val"] == 1
    assert args["tr_num_workers"] == 0


def test_nisqa_passes_checkpoint_compatibility_defaults_to_local_model(monkeypatch, tmp_path):
    import core.stereo_benchmark.models as models

    checkpoint = tmp_path / "nisqa.tar"
    checkpoint.write_bytes(b"checkpoint")
    seen = {}

    class FakeNisqaModel:
        def __init__(self, args):
            seen.update(args)

        def predict(self):
            return {"mos_pred": 3.5}

    monkeypatch.setenv("NISQA_MODEL_PATH", str(checkpoint))
    monkeypatch.setattr(models, "_ensure_nisqa_installed", lambda: True)
    monkeypatch.setattr(models, "_load_nisqa_model_class", lambda: FakeNisqaModel)

    result = models._nisqa_one(np.ones(16000, dtype=np.float32), "cpu")

    assert result == {"status": "ok", "nisqa_mos": 3.5}
    assert seen["ms_channel"] is None
    assert seen["tr_num_workers"] == 0


def test_model_preflight_explains_torch_torchaudio_abi_mismatch():
    from core.stereo_benchmark.preflight import _model_load_failure

    failure = _model_load_failure(
        "SQUIM local initialization",
        OSError("libtorchaudio.so: undefined symbol: _ZNK5torch8autograd4Node4nameB5cxx11Ev"),
    )

    assert "Torch/Torchaudio ABI mismatch" in failure
    assert "same exact release" in failure


def test_speaker_bundle_requires_local_files_without_model_identifier(monkeypatch, tmp_path):
    import core.stereo_benchmark.models as models

    monkeypatch.setenv("SPEECHBRAIN_MODEL_PATH", str(tmp_path))
    monkeypatch.delenv("DUPLEX_MODEL_DIR", raising=False)

    with pytest.raises(FileNotFoundError, match="hyperparams.yaml"):
        models._speaker_model_path()


def test_benchmark_env_config_exports_server_local_model_paths(monkeypatch, tmp_path):
    from core.stereo_benchmark.runtime import apply_environment_config

    model_root = tmp_path / "duplex-model-dir"
    config = tmp_path / "sever.yaml"
    config.write_text(
        """name: sever
offline: true
paths:
  base_models: %s
  speechbrain: ${env.paths.base_models}/spker
  squim: ${env.paths.base_models}/squim_objective_dns2020.pth
  nisqa: ${env.paths.base_models}/nisqa.tar
  dnsmos: ${env.paths.base_models}/dnsmos
""" % model_root,
        encoding="utf-8",
    )
    for variable in ("SPEECHBRAIN_MODEL_PATH", "SQUIM_MODEL_PATH", "NISQA_MODEL_PATH", "DNSMOS_MODEL_PATH"):
        monkeypatch.delenv(variable, raising=False)

    paths = apply_environment_config(config)

    assert paths["SPEECHBRAIN_MODEL_PATH"] == str(model_root / "spker")
    assert paths["DNSMOS_MODEL_PATH"] == str(model_root / "dnsmos")
    assert __import__("os").environ["SPEECHBRAIN_MODEL_PATH"] == str(model_root / "spker")


def test_speaker_bundle_requires_all_local_ecapa_artifacts(monkeypatch, tmp_path):
    import core.stereo_benchmark.models as models

    monkeypatch.setenv("SPEECHBRAIN_MODEL_PATH", str(tmp_path))
    for name in ("hyperparams.yaml", "embedding_model.ckpt", "classifier.ckpt", "label_encoder.txt"):
        (tmp_path / name).touch()

    with pytest.raises(FileNotFoundError, match="mean_var_norm_emb.ckpt"):
        models._speaker_model_path()


def test_squim_requires_local_weights_without_hub_download(monkeypatch, tmp_path):
    import torch
    import core.stereo_benchmark.models as models

    models._squim_model.cache_clear()
    monkeypatch.delenv("SQUIM_MODEL_PATH", raising=False)
    monkeypatch.delenv("DUPLEX_MODEL_DIR", raising=False)
    monkeypatch.delenv("MODE", raising=False)
    monkeypatch.delenv("PIPELINE_MODE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))

    with pytest.raises(FileNotFoundError, match="SQUIM_MODEL_PATH"):
        models._squim_model("cpu")


def test_check_models_aborts_before_corpus_benchmark(monkeypatch, tmp_path):
    import scripts.benchmark_stereo as command

    monkeypatch.setattr(command, "check_benchmark_models", lambda *_args, **_kwargs: {
        "dnsmos": {"status": "OK", "details": "ready"},
        "speechbrain_ecapa": {"status": "MISSING", "details": "missing"},
        "squim": {"status": "OK", "details": "ready"},
        "nisqa": {"status": "OK", "details": "ready"},
    })
    monkeypatch.setattr(command, "print_model_check_summary", lambda _results: None)
    monkeypatch.setattr(command, "run_corpus_benchmark", lambda *_args: pytest.fail("corpus benchmark must not run"))
    monkeypatch.setattr("sys.argv", ["benchmark_stereo.py", "--check-models", "--corpus", str(tmp_path)])

    with pytest.raises(SystemExit) as exc_info:
        command.main()

    assert exc_info.value.code == 1
