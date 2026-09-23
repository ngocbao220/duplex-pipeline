import json

from core.stereo_benchmark.report import RunningSummary


def test_live_summary_bootstraps_and_accepts_new_complete_reports(tmp_path):
    from scripts.watch_benchmark_summary import update_summary

    summary = RunningSummary()
    seen = set()
    target = tmp_path / "live_corpus_report.json"
    update_summary(tmp_path, summary, seen)
    assert json.loads(target.read_text())["summary"]["sample_count"] == 0

    first = tmp_path / "files" / "000000" / "report.json"
    first.parent.mkdir(parents=True)
    first.write_text(json.dumps({"acoustic_quality": {"dnsmos": {"mean": {"ovrl": 3.0}}}}))
    update_summary(tmp_path, summary, seen)
    assert json.loads(target.read_text())["summary"]["metrics"]["dnsmos_ovrl"] == 3.0

    second = tmp_path / "files" / "000001" / "report.json"
    second.parent.mkdir(parents=True)
    second.write_text("{unfinished")
    update_summary(tmp_path, summary, seen)
    assert json.loads(target.read_text())["summary"]["sample_count"] == 1

    second.write_text(json.dumps({"acoustic_quality": {"dnsmos": {"mean": {"ovrl": 5.0}}}}))
    update_summary(tmp_path, summary, seen)
    update_summary(tmp_path, summary, seen)
    result = json.loads(target.read_text())
    assert result["summary"]["sample_count"] == 2
    assert result["summary"]["metrics"]["dnsmos_ovrl"] == 4.0
