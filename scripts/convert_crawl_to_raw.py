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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from core.runtime_cpu import enforce_single_cpu_thread
    enforce_single_cpu_thread()
except ImportError:
    pass


def convert_audio_to_wav(input_path: Path, output_path: Path) -> bool:
    """Converts any input audio file to 16kHz Mono PCM WAV using ffmpeg.
    
    Uses -threads 1 to strictly limit CPU usage to 1 thread per worker.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-threads", "1",
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


def process_youtube(crawl_dir: Path, raw_dir: Path, workers: int = 4):
    print(f"=== Processing YouTube crawl (workers={workers}) ===")
    youtube_crawl = crawl_dir / "youtube"
    if not youtube_crawl.exists():
        print(f"Directory {youtube_crawl} does not exist. Skipping.")
        return

    webm_files = sorted(list(youtube_crawl.glob("*.webm"))) + sorted(
        list(youtube_crawl.glob("*.m4a"))
    )
    if not webm_files:
        print(f"No audio files found in {youtube_crawl}. Skipping.")
        return

    youtube_raw = raw_dir / "youtube"
    youtube_raw.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(webm_files)} YouTube files in {youtube_crawl}.")

    tasks = []
    for idx, f in enumerate(webm_files, start=1):
        out_name = f"youtube_{idx}.wav"
        out_path = youtube_raw / out_name
        tasks.append((f, out_path, out_name))

    def _worker(item):
        in_f, out_p, name = item
        ok = convert_audio_to_wav(in_f, out_p)
        return ok, in_f, out_p, name

    count = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        for future in as_completed(futures):
            ok, in_f, out_p, name = future.result()
            if ok:
                count += 1
                print(f"  -> Converted {in_f.name} -> {name} ({out_p.stat().st_size} bytes)")

    print(
        f"Finished YouTube: {count}/{len(webm_files)} "
        "converted successfully.\n"
    )


def process_podcast_index(crawl_dir: Path, raw_dir: Path, workers: int = 4):
    print(f"=== Processing Podcast Index crawl (workers={workers}) ===")
    podcast_crawl = crawl_dir / "podcast_index"
    if not podcast_crawl.exists():
        print(f"Directory {podcast_crawl} does not exist. Skipping.")
        return

    tar_files = sorted(list(podcast_crawl.glob("*.tar.gz"))) + sorted(
        list(podcast_crawl.glob("*.tar"))
    )
    if not tar_files:
        print(f"No archive files found in {podcast_crawl}. Skipping.")
        return

    podcast_raw = raw_dir / "podcast_index"
    podcast_raw.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(tar_files)} tar archives in {podcast_crawl}.")

    count = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        for tar_f in tar_files:
            print(f"Extracting audio from archive {tar_f.name}...")

            with tarfile.open(tar_f, "r:*") as tar:
                members = sorted(
                    [
                        m
                        for m in tar.getmembers()
                        if m.name.endswith(".audio.mp3")
                        or m.name.endswith(".mp3")
                    ],
                    key=lambda m: m.name,
                )

                extracted_tasks = []
                for member in members:
                    count += 1
                    out_name = f"podcast_{count}.wav"
                    out_path = podcast_raw / out_name

                    extracted_file = tmp_path / Path(member.name).name
                    with tar.extractfile(member) as src, open(extracted_file, "wb") as dst:
                        dst.write(src.read())

                    extracted_tasks.append((extracted_file, out_path, out_name))

                def _convert_and_cleanup(item):
                    in_f, out_p, name = item
                    ok = convert_audio_to_wav(in_f, out_p)
                    if in_f.exists():
                        try:
                            os.remove(in_f)
                        except Exception:
                            pass
                    return ok, name, out_p

                with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                    futures = {pool.submit(_convert_and_cleanup, t): t for t in extracted_tasks}
                    for future in as_completed(futures):
                        ok, name, out_p = future.result()
                        if ok:
                            print(f"  -> Created {out_p} ({out_p.stat().st_size} bytes)")

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
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of parallel conversion workers (default: 4)",
    )
    parser.add_argument(
        "--youtube",
        action="store_true",
        help="Only process YouTube crawled files",
    )
    parser.add_argument(
        "--podcast-index",
        action="store_true",
        help="Only process Podcast Index crawled files",
    )

    args = parser.parse_args()

    run_all = not args.youtube and not args.podcast_index
    if run_all or args.youtube:
        process_youtube(args.crawl_dir, args.raw_dir, workers=args.workers)
    if run_all or args.podcast_index:
        process_podcast_index(args.crawl_dir, args.raw_dir, workers=args.workers)

    print("Audio conversion complete!")