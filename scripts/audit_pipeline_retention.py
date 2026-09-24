"""Compare split dialogue durations with DuplexChat and/or Sommelier outputs."""

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
        if any(part.startswith(".") or part == "debug" for part in relative.parts):
            continue
        number = path.stem.removeprefix(f"{prefix}_")
        if not number.isdigit():
            continue
        key = (relative.parent, number)
        if key in indexed:
            raise ValueError(f"Duplicate {prefix} index under {root}: {key}")
        indexed[key] = path
    return indexed


def audit(dialogue_root: Path, output_roots: dict[str, Path]) -> tuple[list[dict], dict]:
    dialogues = _index(dialogue_root, "dialogue")
    outputs = {name: _index(root, "stereo") for name, root in output_roots.items()}
    keys = set(dialogues)
    for indexed in outputs.values():
        keys.update(indexed)

    rows = []
    summary = {name: {} for name in outputs}

    def source_summary(name: str, source: str) -> tuple[dict, dict]:
        if source not in summary[name]:
            summary[name][source] = (
                {"dialogue": 0.0, "stereo": 0.0, "all_dialogue": 0.0, "all_stereo": 0.0},
                {"paired": 0, "missing": 0, "orphan": 0, "shorter": 0, "longer": 0},
            )
        return summary[name][source]

    for key in sorted(keys, key=lambda item: (str(item[0]), int(item[1]))):
        source = key[0].parts[0] if key[0].parts else "(root)"
        dialogue = dialogues.get(key)
        dialogue_sec = _audio_seconds(dialogue) if dialogue else None
        row = {
            "source": source,
            "relative_dir": str(key[0]),
            "index": key[1],
            "dialogue_path": str(dialogue) if dialogue else "",
            "dialogue_seconds": dialogue_sec,
        }
        for name, indexed in outputs.items():
            stereo = indexed.get(key)
            stereo_sec = _audio_seconds(stereo) if stereo else None
            difference = stereo_sec - dialogue_sec if stereo_sec is not None and dialogue_sec is not None else None
            status = "ok" if difference is not None else "missing_stereo" if dialogue else "orphan_stereo"
            row.update({
                f"{name}_status": status,
                f"{name}_path": str(stereo) if stereo else "",
                f"{name}_seconds": stereo_sec,
                f"{name}_delta_seconds": difference,
            })
            seconds, counts = source_summary(name, source)
            if dialogue_sec is not None:
                seconds["all_dialogue"] += dialogue_sec
            if stereo_sec is not None:
                seconds["all_stereo"] += stereo_sec
            if status == "ok":
                seconds["dialogue"] += dialogue_sec
                seconds["stereo"] += stereo_sec
                counts["paired"] += 1
                counts["shorter"] += difference < -0.01
                counts["longer"] += difference > 0.01
            elif status == "missing_stereo":
                counts["missing"] += 1
            else:
                counts["orphan"] += 1
        rows.append(row)
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dialogue_dir", type=Path, help="Root containing dialogue_N.wav files")
    parser.add_argument("--duplexchat-dir", type=Path, help="Root containing DuplexChat stereo_N.wav outputs")
    parser.add_argument("--sommelier-dir", type=Path, help="Root containing Sommelier stereo_N.wav outputs")
    parser.add_argument("--output-csv", type=Path, default=Path("pipeline_retention_audit.csv"))
    args = parser.parse_args()
    roots = {name: path for name, path in (
        ("duplexchat", args.duplexchat_dir), ("sommelier", args.sommelier_dir),
    ) if path is not None}
    if not roots:
        parser.error("provide at least one of --duplexchat-dir or --sommelier-dir")
    if not args.dialogue_dir.is_dir() or any(not path.is_dir() for path in roots.values()):
        parser.error("all input directories must exist")

    rows, summary = audit(args.dialogue_dir, roots)
    fields = ["source", "relative_dir", "index", "dialogue_path", "dialogue_seconds"]
    for name in roots:
        fields.extend([f"{name}_status", f"{name}_path", f"{name}_seconds", f"{name}_delta_seconds"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dialogue files: {sum(row['dialogue_path'] != '' for row in rows)}")
    for name, sources in summary.items():
        print(f"{name}:")
        for source, (seconds, counts) in sorted(sources.items()):
            delta = seconds["stereo"] - seconds["dialogue"]
            print(f"  {source}: paired {counts['paired']} | missing {counts['missing']} | orphan {counts['orphan']}")
            print(f"    All files: dialogue {seconds['all_dialogue'] / 3600:.2f} h | stereo {seconds['all_stereo'] / 3600:.2f} h")
            print(f"    Paired duration: {seconds['dialogue'] / 3600:.2f} h -> {seconds['stereo'] / 3600:.2f} h ({delta / 3600:+.2f} h)")
            print(f"    Shorter outputs: {counts['shorter']} | longer outputs: {counts['longer']}")
    print(f"CSV: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
