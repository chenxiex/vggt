#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare DTU evaluation data")
    parser.add_argument("--output", default="data", help="Output directory for extracted files")
    parser.add_argument(
        "--cache",
        default=None,
        help="Cache directory for temporary zip files (default: same as --output)",
    )
    return parser.parse_args()


def get_remote_content_length(url: str) -> int | None:
    """Fetch Content-Length header from remote file."""
    try:
        result = subprocess.run(
            ["curl", "-sI", "-L", url],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            if line.lower().startswith("content-length:"):
                return int(line.split(":")[1].strip())
    except Exception as e:
        print(f"Warning: Could not fetch remote file size: {e}", file=sys.stderr)
    return None


def is_download_complete(url: str, local_path: Path) -> bool:
    """Check if local file matches remote file size."""
    if not local_path.exists():
        return False
    remote_size = get_remote_content_length(url)
    if remote_size is None:
        return False
    local_size = local_path.stat().st_size
    return local_size == remote_size


def download_with_resume(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if is_download_complete(url, output_path):
            print(f"File already complete: {output_path}")
            return
        print(f"Resuming download {url} to {output_path}...")
    else:
        print(f"Downloading {url} to {output_path}...")

    cmd = ["curl", "-L", "-C", "-", "-o", str(output_path), url]
    subprocess.run(cmd, check=True)


def flatten_nested_directory(extract_dir: Path, output_path: Path) -> None:
    nested_dir = extract_dir / output_path.stem
    if not nested_dir.is_dir():
        return

    for item in nested_dir.iterdir():
        target = extract_dir / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(item), str(target))

    try:
        nested_dir.rmdir()
    except OSError:
        pass


def download_and_extract(url: str, zip_path: Path, extract_dir: Path) -> None:
    if extract_dir.is_dir():
        print(f"Directory {extract_dir} already exists. Skipping download and extraction.")
        return

    download_with_resume(url, zip_path)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    flatten_nested_directory(extract_dir, zip_path)


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    cache = Path(args.cache) if args.cache else output
    cache.mkdir(parents=True, exist_ok=True)

    download_and_extract(
        "https://www.kaggle.com/api/v1/datasets/download/chenxiex/dtu-test-1200",
        cache / "dtu-test-1200.zip",
        output / "dtu-test-1200",
    )

    download_and_extract(
        "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip",
        cache / "dtu_depths_raw.zip",
        output / "dtu_depths_raw",
    )

    if not (output / "dtu_sample").is_dir():
        download_and_extract(
            "http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip",
            cache / "dtu_sample.zip",
            output / "dtu_sample",
        )

        points_extract_dir = cache / "dtu_points"
        download_and_extract(
            "http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip",
            cache / "dtu_points.zip",
            points_extract_dir,
        )

        source_stl_dir = points_extract_dir / "Points" / "stl"
        target_stl_dir = output / "dtu_sample" / "SampleSet" / "MVS Data" / "Points" / "stl"
        target_stl_dir.mkdir(parents=True, exist_ok=True)
        for item in source_stl_dir.iterdir():
            shutil.copy2(item, target_stl_dir / item.name)

        shutil.rmtree(points_extract_dir, ignore_errors=True)

    for zip_name in ["dtu-test-1200.zip", "dtu_depths_raw.zip", "dtu_sample.zip", "dtu_points.zip"]:
        zip_file = cache / zip_name
        if zip_file.exists():
            zip_file.unlink()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        raise SystemExit(exc.returncode)
