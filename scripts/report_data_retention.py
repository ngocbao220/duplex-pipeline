"""Script to measure audio data retention (in hours and percentages)
across data processing phases for youtube and podcast_index datasets.
"""

import argparse
from pathlib import Path
import wave

ROOT_DIR = Path(__file__).resolve().parents[1]


def get_wav_duration_seconds(wav_path: Path) -> float:
    """Returns the duration of a WAV file in seconds using standard wave module."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate) if rate > 0 else 0.0
    except Exception:
        return 0.0


def calculate_dir_duration(dir_path: Path, pattern: str = "*.wav") -> tuple[float, int]:
    """Recursively calculates total duration in seconds and count of matching WAV files."""
    if not dir_path.exists():
        return 0.0, 0
    wav_files = [f for f in dir_path.rglob(pattern) if not f.name.startswith(".")]
    total_sec = sum(get_wav_duration_seconds(f) for f in wav_files)
    return total_sec, len(wav_files)


def format_hours(seconds: float) -> str:
    """Formats seconds into hours string with 2 decimal places."""
    hours = seconds / 3600.0
    return f"{hours:.2f} h"


def format_pct(numerator: float, denominator: float) -> str:
    """Formats percentage string with 1 decimal place."""
    if denominator <= 0:
        return "N/A"
    pct = (numerator / denominator) * 100.0
    return f"{pct:.1f}%"


def resolve_raw_dataset_dir(raw_dir: Path, ds: str) -> Path:
    """Find the raw dataset directory matching ds with fallbacks for podcast_index vs podcast-index vs podcast."""
    candidates = [
        raw_dir / ds,
        raw_dir / ds.replace("_", "-"),
    ]
    if "_" in ds:
        candidates.append(raw_dir / ds.split("_")[0])
    if raw_dir.name in (ds, ds.replace("_", "-"), ds.split("_")[0]):
        candidates.append(raw_dir)
    if raw_dir.parent.exists():
        candidates.append(raw_dir.parent / ds)
        candidates.append(raw_dir.parent / ds.replace("_", "-"))

    for cand in candidates:
        if cand.exists() and len(list(cand.rglob("*.wav"))) > 0:
            return cand

    for cand in candidates:
        if cand.exists():
            return cand

    return raw_dir / ds


def format_ascii_table(
    datasets: list[str],
    stats: dict,
    total_raw_sec: float,
    total_dialogue_sec: float,
    total_duplex_sec: float,
    total_sommelier_sec: float,
) -> str:
    """Renders a clean ASCII box table (nvidia-smi style grid) for terminal output."""
    headers = [
        "Dataset", "Phase 0 (Raw)", "Phase 1 (Dialogue)", "P1/Raw %",
        "Phase 2A (Duplex)", "Duplex/Raw %", "Phase 2B (Somm)", "Somm/Raw %"
    ]
    rows = []
    for ds in datasets:
        s = stats[ds]
        rows.append([
            ds,
            f"{format_hours(s['raw_sec'])} ({s['raw_count']} files)",
            f"{format_hours(s['dialogue_sec'])} ({s['dialogue_count']} files)",
            format_pct(s['dialogue_sec'], s['raw_sec']),
            f"{format_hours(s['duplex_sec'])} ({s['duplex_count']} files)",
            format_pct(s['duplex_sec'], s['raw_sec']),
            f"{format_hours(s['sommelier_sec'])} ({s['sommelier_count']} files)",
            format_pct(s['sommelier_sec'], s['raw_sec']),
        ])
    total_row = [
        "TOTAL",
        format_hours(total_raw_sec),
        format_hours(total_dialogue_sec),
        format_pct(total_dialogue_sec, total_raw_sec),
        format_hours(total_duplex_sec),
        format_pct(total_duplex_sec, total_raw_sec),
        format_hours(total_sommelier_sec),
        format_pct(total_sommelier_sec, total_raw_sec),
    ]

    all_rows = [headers] + rows + [total_row]
    col_widths = [max(len(row[i]) for row in all_rows) for i in range(len(headers))]

    def make_border():
        return "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    border = make_border()
    lines = [
        border,
        "| " + " | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers))) + " |",
        border,
    ]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[i]:<{col_widths[i]}}" for i in range(len(r))) + " |")
    lines.append(border)
    lines.append("| " + " | ".join(f"{total_row[i]:<{col_widths[i]}}" for i in range(len(total_row))) + " |")
    lines.append(border)
    return "\n".join(lines)


def generate_retention_report(
    raw_dir: Path,
    processed_dir: Path,
    output_md: Path | None = None,
) -> str:
    datasets = ["youtube", "podcast_index"]
    stats = {}

    total_raw_sec = 0.0
    total_dialogue_sec = 0.0
    total_duplex_sec = 0.0
    total_sommelier_sec = 0.0

    for ds in datasets:
        ds_raw_dir = resolve_raw_dataset_dir(raw_dir, ds)

        ds_dialogue_dir = processed_dir / "dialogue" / ds
        ds_duplex_dir = processed_dir / "duplexchat" / ds
        ds_sommelier_dir = processed_dir / "sommelier" / ds

        raw_sec, raw_count = calculate_dir_duration(ds_raw_dir, pattern="*.wav")
        dialogue_sec, dialogue_count = calculate_dir_duration(ds_dialogue_dir, pattern="*.wav")
        duplex_sec, duplex_count = calculate_dir_duration(ds_duplex_dir, pattern="*.wav")
        sommelier_sec, sommelier_count = calculate_dir_duration(ds_sommelier_dir, pattern="*.wav")

        total_raw_sec += raw_sec
        total_dialogue_sec += dialogue_sec
        total_duplex_sec += duplex_sec
        total_sommelier_sec += sommelier_sec

        stats[ds] = {
            "raw_sec": raw_sec,
            "raw_count": raw_count,
            "dialogue_sec": dialogue_sec,
            "dialogue_count": dialogue_count,
            "duplex_sec": duplex_sec,
            "duplex_count": duplex_count,
            "sommelier_sec": sommelier_sec,
            "sommelier_count": sommelier_count,
        }

    # Build Markdown Report
    lines = [
        "# Data Retention & Pipeline Performance Report",
        "",
        "Báo cáo thống kê tổng thời lượng âm thanh (tính bằng Giờ) và tỉ lệ giữ lại dữ liệu (%) qua từng giai đoạn xử lý.",
        "",
        "## Summary Table",
        "",
        "| Source Dataset | Phase 0: Raw (Hours) | Phase 1: Dialogue (Hours) | Retention % (Phase 1 / Raw) | Phase 2A: DuplexChat (Hours) | Retention % (DuplexChat / Raw) | Phase 2B: Sommelier (Hours) | Retention % (Sommelier / Raw) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for ds in datasets:
        s = stats[ds]
        lines.append(
            f"| **{ds}** | {format_hours(s['raw_sec'])} ({s['raw_count']} files) "
            f"| {format_hours(s['dialogue_sec'])} ({s['dialogue_count']} files) "
            f"| {format_pct(s['dialogue_sec'], s['raw_sec'])} "
            f"| {format_hours(s['duplex_sec'])} ({s['duplex_count']} files) "
            f"| {format_pct(s['duplex_sec'], s['raw_sec'])} "
            f"| {format_hours(s['sommelier_sec'])} ({s['sommelier_count']} files) "
            f"| {format_pct(s['sommelier_sec'], s['raw_sec'])} |"
        )

    # Add Total Row
    lines.append(
        f"| **TOTAL** | **{format_hours(total_raw_sec)}** "
        f"| **{format_hours(total_dialogue_sec)}** "
        f"| **{format_pct(total_dialogue_sec, total_raw_sec)}** "
        f"| **{format_hours(total_duplex_sec)}** "
        f"| **{format_pct(total_duplex_sec, total_raw_sec)}** "
        f"| **{format_hours(total_sommelier_sec)}** "
        f"| **{format_pct(total_sommelier_sec, total_raw_sec)}** |"
    )

    lines.extend([
        "",
        "## Phase Definitions",
        "- **Phase 0 (Raw)**: Dữ liệu âm thanh thô ban đầu (PCM 16kHz mono).",
        "- **Phase 1 (Dialogue Extraction)**: Các đoạn hội thoại có đúng 2 speaker sau khi lọc ngắt câu và lọc nhạc bằng Demucs.",
        "- **Phase 2A (DuplexChat Separation)**: Audio stereo 24kHz sau khi chạy mô hình DialogueSidon.",
        "- **Phase 2B (Sommelier Separation)**: Audio stereo 24kHz sau khi tách overlap bằng SepReFormer và khớp Cosine Embedding.",
        ""
    ])

    report_content = "\n".join(lines)

    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(report_content, encoding="utf-8")
        print(f"Report written to: {output_md}")

    return report_content


def print_rich_table(
    datasets: list[str],
    stats: dict,
    total_raw_sec: float,
    total_dialogue_sec: float,
    total_duplex_sec: float,
    total_sommelier_sec: float,
) -> None:
    """Renders a styled Rich Table for terminal output with fallback to ASCII table."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Data Retention & Pipeline Performance Report", header_style="bold cyan")

        table.add_column("Source Dataset", style="bold white", justify="left")
        table.add_column("Phase 0: Raw", justify="right", style="green")
        table.add_column("Phase 1: Dialogue", justify="right", style="yellow")
        table.add_column("Retention (P1/Raw)", justify="right", style="magenta")
        table.add_column("Phase 2A: DuplexChat", justify="right", style="blue")
        table.add_column("Retention (Duplex/Raw)", justify="right", style="magenta")
        table.add_column("Phase 2B: Sommelier", justify="right", style="cyan")
        table.add_column("Retention (Somm/Raw)", justify="right", style="magenta")

        for ds in datasets:
            s = stats[ds]
            table.add_row(
                ds,
                f"{format_hours(s['raw_sec'])} ({s['raw_count']} files)",
                f"{format_hours(s['dialogue_sec'])} ({s['dialogue_count']} files)",
                format_pct(s['dialogue_sec'], s['raw_sec']),
                f"{format_hours(s['duplex_sec'])} ({s['duplex_count']} files)",
                format_pct(s['duplex_sec'], s['raw_sec']),
                f"{format_hours(s['sommelier_sec'])} ({s['sommelier_count']} files)",
                format_pct(s['sommelier_sec'], s['raw_sec']),
            )

        table.add_section()
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{format_hours(total_raw_sec)}[/bold]",
            f"[bold]{format_hours(total_dialogue_sec)}[/bold]",
            f"[bold]{format_pct(total_dialogue_sec, total_raw_sec)}[/bold]",
            f"[bold]{format_hours(total_duplex_sec)}[/bold]",
            f"[bold]{format_pct(total_duplex_sec, total_raw_sec)}[/bold]",
            f"[bold]{format_hours(total_sommelier_sec)}[/bold]",
            f"[bold]{format_pct(total_sommelier_sec, total_raw_sec)}[/bold]",
        )

        console.print(table)
    except ImportError:
        ascii_table = format_ascii_table(datasets, stats, total_raw_sec, total_dialogue_sec, total_duplex_sec, total_sommelier_sec)
        print("\n" + ascii_table)


def main():
    parser = argparse.ArgumentParser(description="Generate data retention report across processing phases.")
    parser.add_argument("--raw-dir", type=Path, default=ROOT_DIR / "data" / "raw",
                        help="Path to raw data directory.")
    parser.add_argument("--processed-dir", type=Path, default=ROOT_DIR / "data" / "processed",
                        help="Path to processed data directory.")
    parser.add_argument("--output-md", type=Path, default=ROOT_DIR / "outputs" / "data_retention_report.md",
                        help="Output path for markdown report.")
    args = parser.parse_args()

    raw_path = args.raw_dir.resolve()
    processed_path = args.processed_dir.resolve()
    output_md_path = args.output_md.resolve()

    report_md = generate_retention_report(raw_path, processed_path, output_md_path)

    # Calculate stats for terminal table display
    datasets = ["youtube", "podcast_index"]
    stats = {}
    total_raw_sec = total_dialogue_sec = total_duplex_sec = total_sommelier_sec = 0.0

    for ds in datasets:
        ds_raw_dir = resolve_raw_dataset_dir(raw_path, ds)
        ds_dialogue_dir = processed_path / "dialogue" / ds
        ds_duplex_dir = processed_path / "duplexchat" / ds
        ds_sommelier_dir = processed_path / "sommelier" / ds

        raw_sec, raw_count = calculate_dir_duration(ds_raw_dir, pattern="*.wav")
        dialogue_sec, dialogue_count = calculate_dir_duration(ds_dialogue_dir, pattern="*.wav")
        duplex_sec, duplex_count = calculate_dir_duration(ds_duplex_dir, pattern="*.wav")
        sommelier_sec, sommelier_count = calculate_dir_duration(ds_sommelier_dir, pattern="*.wav")

        total_raw_sec += raw_sec
        total_dialogue_sec += dialogue_sec
        total_duplex_sec += duplex_sec
        total_sommelier_sec += sommelier_sec

        stats[ds] = {
            "raw_sec": raw_sec,
            "raw_count": raw_count,
            "dialogue_sec": dialogue_sec,
            "dialogue_count": dialogue_count,
            "duplex_sec": duplex_sec,
            "duplex_count": duplex_count,
            "sommelier_sec": sommelier_sec,
            "sommelier_count": sommelier_count,
        }

    print_rich_table(datasets, stats, total_raw_sec, total_dialogue_sec, total_duplex_sec, total_sommelier_sec)


if __name__ == "__main__":
    main()
