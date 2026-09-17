"""Script to convert crawled YouTube (.webm/.m4a) and Podcast Index (.tar.gz/.tar)
files into standard PCM 16kHz WAV format in data/raw/.

Naming scheme:

- data/raw/youtube/youtube_1.wav, youtube_2.wav, ...
- data/raw/podcast_index/podcast_1.wav, podcast_2.wav, ...
"""

import argparse
import os
import tarfile
import tempfile
import subprocess
from pathlib import Path


def convert_audio_to_wav(input_path: Path, output_path: Path) -> bool:
    """Converts any input audio file to 16kHz Mono PCM WAV using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print(
            f"Error converting {input_path}:\n"
            f"{res.stderr.decode('utf-8', errors='ignore')}"
        )
        return False
    return True


def process_youtube(crawl_dir: Path, raw_dir: Path):
    print("=== Processing YouTube crawl ===")
    youtube_crawl = crawl_dir / "youtube"
    youtube_raw = raw_dir / "youtube"
    youtube_raw.mkdir(parents=True, exist_ok=True)

    webm_files = sorted(list(youtube_crawl.glob("*.webm"))) + sorted(
        list(youtube_crawl.glob("*.m4a"))
    )
    print(f"Found {len(webm_files)} YouTube files in {youtube_crawl}.")

    count = 0
    for idx, f in enumerate(webm_files, start=1):
        out_name = f"youtube_{idx}.wav"
        out_path = youtube_raw / out_name
        print(f"Converting {f.name} -> {out_name}...")
        if convert_audio_to_wav(f, out_path):
            count += 1
            print(f"  -> Created {out_path} ({out_path.stat().st_size} bytes)")

    print(
        f"Finished YouTube: {count}/{len(webm_files)} "
        "converted successfully.\n"
    )


def process_podcast_index(crawl_dir: Path, raw_dir: Path):
    print("=== Processing Podcast Index crawl ===")
    podcast_crawl = crawl_dir / "podcast_index"
    podcast_raw = raw_dir / "podcast_index"
    podcast_raw.mkdir(parents=True, exist_ok=True)

    tar_files = sorted(list(podcast_crawl.glob("*.tar.gz"))) + sorted(
        list(podcast_crawl.glob("*.tar"))
    )
    print(f"Found {len(tar_files)} tar archives in {podcast_crawl}.")

    count = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        for tar_f in tar_files:
            print(f"Extracting audio from archive {tar_f.name}...")

            with tarfile.open(tar_f, "r:*") as tar:
                # Sort members deterministically by name
                members = sorted(
                    [
                        m
                        for m in tar.getmembers()
                        if m.name.endswith(".audio.mp3")
                        or m.name.endswith(".mp3")
                    ],
                    key=lambda m: m.name,
                )

                for member in members:
                    count += 1
                    out_name = f"podcast_{count}.wav"
                    out_path = podcast_raw / out_name

                    extracted_file = tmp_path / Path(member.name).name
                    with tar.extractfile(member) as src, open(
                        extracted_file, "wb"
                    ) as dst:
                        dst.write(src.read())

                    print(f"Converting {extracted_file.name} -> {out_name}...")

                    if convert_audio_to_wav(extracted_file, out_path):
                        print(
                            f"  -> Created {out_path} "
                            f"({out_path.stat().st_size} bytes)"
                        )
                    else:
                        count -= 1

                    if extracted_file.exists():
                        os.remove(extracted_file)

    print(
        f"Finished Podcast Index: {count} mp3 files "
        "converted successfully.\n"
    )


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    default_crawl_dir = root_dir / "data" / "crawl"
    default_raw_dir = root_dir / "data" / "raw"

    parser = argparse.ArgumentParser(
        description=(
            "Convert crawled audio files into standard PCM "
            "16kHz WAV format."
        )
    )
    parser.add_argument(
        "--crawl-dir",
        type=Path,
        default=default_crawl_dir,
        help=(
            "Path to the root crawl directory containing "
            "youtube/ and podcast_index/"
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=default_raw_dir,
        help="Path to the root raw output directory",
    )

    args = parser.parse_args()

    process_youtube(args.crawl_dir, args.raw_dir)
    process_podcast_index(args.crawl_dir, args.raw_dir)

    print("Step 1 raw data generation complete!")