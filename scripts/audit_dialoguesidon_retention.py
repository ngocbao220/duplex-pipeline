"""Compare split dialogue WAVs with their DialogueSidon stereo outputs."""

import argparse
import csv
import wave
from pathlib import Path


def _audio_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        rate = audio.getframerate()
        if rate <= 0:
            raise ValueError(f"Invalid sample rate in {path}")
        return audio.getnframes() / rate


def _index(root: Path, prefix: str) -> dict[tuple[Path, str], Path]:
    indexed = {}
    for path in root.rglob(f"{prefix}_*.wav"):
        relative = path.relative_to(root)
        if any(part.startswith(".") or part in {"debug", ".history", ".runs"} for part in relative.parts):
            continue
        number = path.stem.removeprefix(f"{prefix}_")
        if not number.isdigit():
            continue
        key = (relative.parent, number)
        if key in indexed:
            raise ValueError(f"Duplicate {prefix} index under {root}: {key}")
        indexed[key] = path
    return indexed


def audit(dialogue_root: Path, stereo_root: Path) -> list[dict]:
    dialogues = _index(dialogue_root, "dialogue")
    stereos = _index(stereo_root, "stereo")
    rows = []
    for key in sorted(dialogues.keys() | stereos.keys(), key=lambda item: (str(item[0]), int(item[1]))):
        dialogue = dialogues.get(key)
        stereo = stereos.get(key)
        dialogue_sec = _audio_seconds(dialogue) if dialogue else None
        stereo_sec = _audio_seconds(stereo) if stereo else None
        difference = stereo_sec - dialogue_sec if dialogue_sec is not None and stereo_sec is not None else None
        status = "ok" if difference is not None else "missing_stereo" if dialogue else "orphan_stereo"
        rows.append({
            "relative_dir": str(key[0]),
            "index": key[1],
            "status": status,
            "dialogue_path": str(dialogue) if dialogue else "",
            "stereo_path": str(stereo) if stereo else "",
            "dialogue_seconds": dialogue_sec,
            "stereo_seconds": stereo_sec,
            "difference_seconds": difference,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dialogue_dir", type=Path, help="Root containing dialogue_N.wav files")
    parser.add_argument("stereo_dir", type=Path, help="Root containing matching stereo_N.wav files")
    parser.add_argument("--output-csv", type=Path, default=Path("dialoguesidon_retention_audit.csv"))
    args = parser.parse_args()
    if not args.dialogue_dir.is_dir() or not args.stereo_dir.is_dir():
        parser.error("dialogue_dir and stereo_dir must both be existing directories")

    rows = audit(args.dialogue_dir, args.stereo_dir)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["relative_dir", "index", "status", "dialogue_path", "stereo_path", "dialogue_seconds", "stereo_seconds", "difference_seconds"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    paired = [row for row in rows if row["status"] == "ok"]
    missing = sum(row["status"] == "missing_stereo" for row in rows)
    orphan = sum(row["status"] == "orphan_stereo" for row in rows)
    dialogue_total = sum(row["dialogue_seconds"] or 0 for row in paired)
    stereo_total = sum(row["stereo_seconds"] or 0 for row in paired)
    shorter = sum(row["difference_seconds"] < -0.01 for row in paired)
    longer = sum(row["difference_seconds"] > 0.01 for row in paired)
    print(f"Paired files: {len(paired)} | missing stereo: {missing} | orphan stereo: {orphan}")
    print(f"Paired duration: dialogue {dialogue_total / 3600:.2f} h -> stereo {stereo_total / 3600:.2f} h")
    print(f"Duration delta: {(stereo_total - dialogue_total) / 3600:.2f} h | shorter outputs: {shorter} | longer outputs: {longer}")
    print(f"CSV: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
