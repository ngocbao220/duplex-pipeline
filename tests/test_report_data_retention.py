import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.report_data_retention import generate_retention_report, get_wav_duration_seconds


def create_dummy_wav(path: Path, duration_sec: float = 1.0, rate: int = 16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = int(duration_sec * rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * num_frames)


def test_retention_report_calculation(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    output_md = tmp_path / "outputs" / "report.md"

    # Create 3600 seconds (1.0 hour) raw YouTube wav
    create_dummy_wav(raw_dir / "youtube" / "youtube_1.wav", duration_sec=3600.0)

    # Create 1800 seconds (0.5 hour) dialogue wav
    create_dummy_wav(processed_dir / "dialogue" / "youtube" / "youtube_1" / "dialogue_1.wav", duration_sec=1800.0)

    # Create 1800 seconds (0.5 hour) duplexchat stereo wav
    create_dummy_wav(processed_dir / "duplexchat" / "youtube" / "youtube_1" / "stereo_1.wav", duration_sec=1800.0)

    report_text = generate_retention_report(raw_dir, processed_dir, output_md)

    assert output_md.is_file()
    assert "1.00 h" in report_text
    assert "0.50 h" in report_text
    assert "50.0%" in report_text


def test_retention_report_podcast_index_fallback(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    output_md = tmp_path / "outputs" / "report.md"

    # Create raw wav in podcast-index (hyphenated)
    create_dummy_wav(raw_dir / "podcast-index" / "podcast_1.wav", duration_sec=3600.0)

    report_text = generate_retention_report(raw_dir, processed_dir, output_md)

    assert "podcast_index" in report_text
    assert "1.00 h" in report_text

