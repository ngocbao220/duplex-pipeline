"""Aggregate and render reference-free stereo benchmark reports."""
from __future__ import annotations


def summarize_reports(reports: list[dict]) -> dict:
    values = {
        "dnsmos_ovrl": [_at(report, "acoustic_quality", "dnsmos", "mean", "ovrl") for report in reports],
        "nisqa_mos": [_at(report, "acoustic_quality", "nisqa", "mean", "nisqa_mos") for report in reports],
        "sq_stoi": [_at(report, "acoustic_quality", "squim", "mean", "sq_stoi") for report in reports],
        "sq_pesq": [_at(report, "acoustic_quality", "squim", "mean", "sq_pesq") for report in reports],
        "itc": [_at(report, "speaker_identity", "itc", "mean", "itc") for report in reports],
        "itd": [_at(report, "speaker_identity", "itd", "itd") for report in reports],
        "overlap_pct": [_at(report, "speech_activity", "overlap", "percentage") for report in reports],
        "turn_exchanges_per_min": [_at(report, "turn_taking", "turn_exchanges_per_min") for report in reports],
        "overlapping_transition_rate": [_at(report, "turn_taking", "overlapping_transition_rate") for report in reports],
        "backchannels_per_min": [_at(report, "turn_taking", "backchannels_per_min") for report in reports],
        "leakage_proxy_db": [
            _mean([
                _at(report, "leakage_proxy", "left_to_right", "median_db"),
                _at(report, "leakage_proxy", "right_to_left", "median_db"),
            ])
            for report in reports
        ],
    }
    metrics = {name: _mean(items) for name, items in values.items()}
    metrics["overlap_transition_pct"] = _scale(metrics.pop("overlapping_transition_rate"), 100)
    return {"sample_count": len(reports), "metrics": metrics}


def render_tables(summary: dict) -> str:
    """Render the requested compact Metric / Value / Range / Target / Meaning table."""
    metrics = summary["metrics"]
    rows = [
        ("DNSMOS", metrics["dnsmos_ovrl"], "[1.0, 5.0]", "->", "Acoustic quality (DNSMOS)", ""),
        ("NISQA-MOS", metrics["nisqa_mos"], "[1.0, 5.0]", "->", "Speech quality MOS (NISQA)", ""),
        ("SQ-STOI", metrics["sq_stoi"], "[0.0, 1.0]", "->", "Estimated intelligibility", ""),
        ("SQ-PESQ", metrics["sq_pesq"], "[-0.5, 4.5]", "->", "Estimated perceptual quality", ""),
        ("ITC", metrics["itc"], "[-1.0, 1.0]", "->", "Intra-track speaker consistency", ""),
        ("ITD", metrics["itd"], "[0.0, 2.0]", "->", "Inter-track distinctiveness", ""),
        ("Overlap", metrics["overlap_pct"], "[0%, 100%]", "-", "Simultaneous speech", " %"),
        ("Backchannel", metrics["backchannels_per_min"], ">= 0", "-", "Estimated short responses (VAD-based)", " /min"),
        ("Turn Exchange", metrics["turn_exchanges_per_min"], ">= 0", "-", "Speaker turn dynamics", " /min"),
        ("Overlap Transition", metrics["overlap_transition_pct"], "[0%, 100%]", "-", "Turns initiated during overlap", " %"),
        ("Leakage Proxy", metrics["leakage_proxy_db"], "(-inf, 0]", "<-", "Inactive-channel energy (rò giọng)", " dB"),
    ]
    formatted = [["Metric", "Value", "Range", "Target", "Meaning"]] + [
        [name, f"{_format(value)}{suffix if value is not None else ''}", val_range, target, meaning]
        for name, value, val_range, target, meaning, suffix in rows
    ]
    return f"Samples scored: {summary['sample_count']}\n\n{_table(formatted)}"


def render_warnings(reports: list[dict]) -> str:
    if not reports:
        return ""
    report = reports[0]
    warnings = []
    
    def check(path_tuple, name):
        data = report
        for key in path_tuple:
            if not isinstance(data, dict):
                return
            data = data.get(key)
        if isinstance(data, dict) and data.get("status") == "unavailable":
            reason = data.get("reason", "Unknown reason")
            warnings.append(f"- {name} skipped: {reason}")

    check(("acoustic_quality", "dnsmos", "left"), "DNSMOS")
    check(("acoustic_quality", "nisqa", "left"), "NISQA")
    check(("acoustic_quality", "squim", "left"), "SQUIM")
    check(("speaker_identity", "itc", "left"), "Speaker Identity (ITC/ITD)")

    if warnings:
        return "\nWarnings:\n" + "\n".join(warnings) + "\n"
    return ""


def flatten_report(report: dict, source: str) -> dict:
    return {"source": source, "status": "ok", "duration_sec": _at(report, "input", "duration_sec"), **summarize_reports([report])["metrics"]}


def _at(data: dict, *keys: str):
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return float(data) if isinstance(data, (int, float)) else None


def _mean(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None


def _format(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _scale(value: float | None, multiplier: float) -> float | None:
    return None if value is None else value * multiplier


def _table(rows: list[list[str]]) -> str:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    def line(row: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return "\n".join([line(rows[0]), separator, *(line(row) for row in rows[1:])])
