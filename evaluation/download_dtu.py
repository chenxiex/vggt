#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile


@dataclass
class DatasetSpec:
    extract_dir: str  # subdirectory name under output
    zip_name: str     # local zip filename (non-MS)
    url: str          # direct download URL (non-MS)
    ms_repo: str      # ModelScope repository (owner/name)
    ms_file: str      # filename within the ModelScope repository


DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        extract_dir="dtu-test-1200",
        zip_name="dtu-test-1200.zip",
        url="https://www.kaggle.com/api/v1/datasets/download/chenxiex/dtu-test-1200",
        ms_repo="anlorsp/dtu-test-1200",
        ms_file="dtu-test-1200.zip",
    ),
    DatasetSpec(
        extract_dir="dtu-depths-raw",
        zip_name="dtu-depths-raw.zip",
        url="https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip",
        ms_repo="anlorsp/dtu-depths-raw",
        ms_file="Depths_raw.zip",
    ),
    DatasetSpec(
        extract_dir="dtu-sample",
        zip_name="dtu-sample.zip",
        url="http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip",
        ms_repo="anlorsp/dtu-sample",
        ms_file="dtu-sample.zip",
    ),
]

# Only needed for non-MS path: Points are already merged in the MS sample dataset
_POINTS_SPEC = DatasetSpec(
    extract_dir="dtu-points",
    zip_name="dtu-points.zip",
    url="http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip",
    ms_repo="",
    ms_file="",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare DTU evaluation data")
    parser.add_argument("--output", default="data", help="Output directory for extracted files")
    parser.add_argument(
        "--cache",
        default=None,
        help="Cache directory for temporary zip files (default: same as --output)",
    )
    parser.add_argument(
        "--ms",
        action="store_true",
        help="Download from ModelScope mirror instead of original sources",
    )
    return parser.parse_args()


def get_remote_content_length(url: str) -> int | None:
    """Fetch Content-Length header from remote URL."""
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
    """Check if local file size matches the remote Content-Length."""
    if not local_path.exists():
        return False
    remote_size = get_remote_content_length(url)
    if remote_size is None:
        return False
    return local_path.stat().st_size == remote_size


def download_with_resume(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if is_download_complete(url, output_path):
            print(f"File already complete: {output_path}")
            return
        print(f"Resuming download {url} to {output_path}...")
    else:
        print(f"Downloading {url} to {output_path}...")
    subprocess.run(["curl", "-L", "-C", "-", "-o", str(output_path), url], check=True)


def download_from_modelscope(ms_repo: str, ms_file: str, cache: Path) -> Path:
    """Download a file from ModelScope to cache and return its local path."""
    local_zip = cache / Path(ms_file).name
    if local_zip.exists():
        print(f"File already in cache: {local_zip}")
        return local_zip
    print(f"Downloading {ms_repo}/{ms_file} from ModelScope...")
    subprocess.run(
        ["modelscope", "download", "--dataset", ms_repo, "--include", ms_file, "--local_dir", str(cache)],
        check=True,
    )
    return local_zip


def get_zip_path(spec: DatasetSpec, cache: Path, use_ms: bool) -> Path:
    """Return the expected local zip path for a dataset spec."""
    if use_ms:
        return cache / Path(spec.ms_file).name
    return cache / spec.zip_name


def acquire_zip(spec: DatasetSpec, cache: Path, use_ms: bool) -> Path:
    """Download zip file if needed and return its local path."""
    if use_ms:
        return download_from_modelscope(spec.ms_repo, spec.ms_file, cache)
    zip_path = cache / spec.zip_name
    download_with_resume(spec.url, zip_path)
    return zip_path


def flatten_nested_directory(extract_dir: Path, zip_path: Path) -> None:
    nested_dir = extract_dir / zip_path.stem
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


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract zip and flatten top-level nested directory if present."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name}...")
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    flatten_nested_directory(extract_dir, zip_path)
    print(f"Extraction complete: {extract_dir}")


def process_dataset(spec: DatasetSpec, output: Path, cache: Path, use_ms: bool) -> None:
    """Download and extract a dataset, skipping if already present."""
    extract_dir = output / spec.extract_dir
    if extract_dir.is_dir():
        print(f"Directory {extract_dir} already exists. Skipping download and extraction.")
        return
    zip_path = acquire_zip(spec, cache, use_ms)
    extract_zip(zip_path, extract_dir)
    print(f"Deleting {zip_path.name}...")
    zip_path.unlink(missing_ok=True)
    print(f"Dataset {spec.extract_dir} processing complete.")


def _merge_points_into_sample(sample_dir: Path, cache: Path) -> None:
    """Download Points.zip, copy stl files into dtu-sample, then clean up."""
    points_dir = cache / _POINTS_SPEC.extract_dir
    points_zip = acquire_zip(_POINTS_SPEC, cache, use_ms=False)
    extract_zip(points_zip, points_dir)
    print(f"Merging Points data into sample directory...")
    source_stl = points_dir / "Points" / "stl"
    target_stl = sample_dir / "SampleSet" / "MVS Data" / "Points" / "stl"
    target_stl.mkdir(parents=True, exist_ok=True)
    for item in source_stl.iterdir():
        shutil.copy2(item, target_stl / item.name)
    shutil.rmtree(points_dir, ignore_errors=True)
    print(f"Deleting {points_zip.name}...")
    points_zip.unlink(missing_ok=True)
    print(f"Points merge complete.")


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    cache = Path(args.cache) if args.cache else output
    cache.mkdir(parents=True, exist_ok=True)

    *regular_specs, sample_spec = DATASETS
    for spec in regular_specs:
        process_dataset(spec, output, cache, use_ms=args.ms)

    # dtu-sample: non-MS mode requires merging a separate Points.zip into the sample dir
    sample_dir = output / sample_spec.extract_dir
    if sample_dir.is_dir():
        print(f"Directory {sample_dir} already exists. Skipping download and extraction.")
    else:
        sample_zip = acquire_zip(sample_spec, cache, use_ms=args.ms)
        extract_zip(sample_zip, sample_dir)
        print(f"Deleting {sample_zip.name}...")
        sample_zip.unlink(missing_ok=True)
        if not args.ms:
            _merge_points_into_sample(sample_dir, cache)

    print("All datasets processed successfully!")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        raise SystemExit(exc.returncode)
