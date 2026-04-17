import argparse
import random
from contextlib import nullcontext
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.request import urlretrieve

import torch

from vggt.models.vggt import VGGT
from vggt.quantization.smoothquant import calibrate_attention_scales, save_smoothquant_artifact
from vggt.utils.load_fn import load_and_preprocess_images

HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co")
MODEL_URL = f"{HF_ENDPOINT}/facebook/VGGT-1B/resolve/main/model.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate SmoothQuant scales for VGGT attention layers")
    parser.add_argument("--model_path", type=Path, default=Path("ckpt/model.pt"), help="Path to model checkpoint")
    parser.add_argument(
        "--calib_dir",
        type=Path,
        default=None,
        help="Calibration image directory. Required when --dtu_test_1200_path is not set.",
    )
    parser.add_argument(
        "--dtu_test_1200_path",
        type=Path,
        default=None,
        help="Path to DTU test-1200 root directory (contains Rectified/ and scan_list_test.txt).",
    )
    parser.add_argument(
        "--dtu_scans",
        type=str,
        default=None,
        help="DTU scene ids for calibration, e.g. 1,2,3. If empty or true, uses scan_list_test.txt.",
    )
    parser.add_argument(
        "--dtu_images_per_scene",
        type=int,
        default=0,
        help="Max number of calibration images sampled per DTU scene. <=0 means use all available images.",
    )
    parser.add_argument("--output_path", type=Path, required=True, help="Output path of SmoothQuant artifact (.pt)")
    parser.add_argument("--num_samples", type=int, default=32, help="Maximum number of calibration images")
    parser.add_argument("--batch_size", type=int, default=8, help="Calibration batch size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for calibration sampling")
    parser.add_argument("--preprocess_mode", type=str, choices=["crop", "pad"], default="crop")
    parser.add_argument("--recursive", action="store_true", help="Recursively search images under calib_dir")
    parser.add_argument("--weight_qmax", type=float, default=127.0, help="Weight quant range max for INT8")
    parser.add_argument("--act_qmax", type=float, default=65504.0, help="Activation range max for A16")
    parser.add_argument("--disable_amp", action="store_true", help="Disable AMP during calibration")
    return parser.parse_args()


def build_dtu_scene_names(dtu_test_1200_path: Path, scans: Optional[str]) -> List[str]:
    if not scans or scans.lower() == "true":
        scan_list_path = dtu_test_1200_path / "scan_list_test.txt"
        if not scan_list_path.exists():
            raise FileNotFoundError(f"DTU scan list not found: {scan_list_path}")
        with open(scan_list_path, encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    scene_ids = [scan_id.strip() for scan_id in scans.split(",") if scan_id.strip()]
    return [f"scan{scene_id}" for scene_id in scene_ids]


def _dtu_scene_seed(scene_name: str, base_seed: int) -> int:
    scene_digits = "".join(ch for ch in scene_name if ch.isdigit())
    if scene_digits:
        return base_seed + int(scene_digits)
    return base_seed + sum(ord(ch) for ch in scene_name)


def collect_dtu_scene_image_paths(
    dtu_test_1200_path: Path,
    scans: Optional[str],
    images_per_scene: int,
    seed: int,
    max_total_images: int,
) -> Dict[str, List[Path]]:
    if not dtu_test_1200_path.exists():
        raise FileNotFoundError(f"DTU root directory not found: {dtu_test_1200_path}")

    scene_names = build_dtu_scene_names(dtu_test_1200_path, scans)
    if not scene_names:
        raise ValueError("No DTU scenes found for calibration")

    scene_to_paths: Dict[str, List[Path]] = {}
    for scene_name in scene_names:
        scene_dir = dtu_test_1200_path / "Rectified" / scene_name
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"DTU scene directory not found: {scene_dir}")

        image_paths = sorted(scene_dir.glob("rect_*_3_r5000.png"))
        if not image_paths:
            raise ValueError(f"No DTU rectified images found in scene: {scene_name}")

        if images_per_scene > 0 and len(image_paths) > images_per_scene:
            rng = random.Random(_dtu_scene_seed(scene_name, seed))
            image_paths = sorted(rng.sample(image_paths, images_per_scene))

        scene_to_paths[scene_name] = image_paths

    total_images = sum(len(paths) for paths in scene_to_paths.values())
    if max_total_images > 0 and total_images > max_total_images:
        # Round-robin truncation keeps scene distribution while capping total samples.
        truncated: Dict[str, List[Path]] = {scene_name: [] for scene_name in scene_names}
        remaining = {scene_name: list(scene_to_paths[scene_name]) for scene_name in scene_names}

        picked = 0
        while picked < max_total_images:
            progressed = False
            for scene_name in scene_names:
                if not remaining[scene_name]:
                    continue
                truncated[scene_name].append(remaining[scene_name].pop(0))
                picked += 1
                progressed = True
                if picked >= max_total_images:
                    break
            if not progressed:
                break

        scene_to_paths = {
            scene_name: paths for scene_name, paths in truncated.items() if paths
        }

    return scene_to_paths


def collect_image_paths(calib_dir: Path, recursive: bool) -> List[Path]:
    if not calib_dir.exists():
        raise FileNotFoundError(f"Calibration directory not found: {calib_dir}")

    image_root = calib_dir / "images" if (calib_dir / "images").is_dir() else calib_dir
    entries = image_root.rglob("*") if recursive else image_root.glob("*")

    image_paths = [
        p for p in entries if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(image_paths)


def sample_image_paths(image_paths: List[Path], num_samples: int, seed: int) -> List[Path]:
    if num_samples <= 0 or len(image_paths) <= num_samples:
        return image_paths
    rng = random.Random(seed)
    return sorted(rng.sample(image_paths, num_samples))


def batched(items: List[Path], batch_size: int) -> Iterable[List[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_state_dict(model_path: Path):
    if not model_path.exists():
        print(f"Checkpoint not found at {model_path}, downloading from {MODEL_URL}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(MODEL_URL, model_path)

    checkpoint = torch.load(model_path, map_location="cpu")
    return checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if args.dtu_images_per_scene < 0:
        raise ValueError("dtu_images_per_scene must be >= 0")

    if args.calib_dir is None and args.dtu_test_1200_path is None:
        raise ValueError("Either --calib_dir or --dtu_test_1200_path must be provided")

    if args.calib_dir is not None and args.dtu_test_1200_path is not None:
        raise ValueError("--calib_dir and --dtu_test_1200_path are mutually exclusive")

    source_type = "dtu" if args.dtu_test_1200_path is not None else "dir"

    if source_type == "dtu":
        scene_to_image_paths = collect_dtu_scene_image_paths(
            dtu_test_1200_path=args.dtu_test_1200_path,
            scans=args.dtu_scans,
            images_per_scene=args.dtu_images_per_scene,
            seed=args.seed,
            max_total_images=args.num_samples,
        )
    else:
        assert args.calib_dir is not None
        image_paths = collect_image_paths(args.calib_dir, recursive=args.recursive)
        if not image_paths:
            raise ValueError(f"No images found in calibration directory: {args.calib_dir}")
        image_paths = sample_image_paths(image_paths, args.num_samples, args.seed)
        scene_to_image_paths = {"custom_scene": image_paths}

    total_images = sum(len(paths) for paths in scene_to_image_paths.values())
    if total_images == 0:
        raise ValueError("No calibration images selected")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if device.type == "cuda" and not args.disable_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print(f"Calibration source: {source_type}")
    print(f"Calibration images: {total_images}")
    print(f"Calibration scenes: {len(scene_to_image_paths)}")
    print(f"Device: {device}")
    if amp_dtype is not None:
        print(f"AMP dtype: {amp_dtype}")

    model = VGGT()
    state_dict = load_state_dict(args.model_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"Unexpected keys when loading checkpoint: {unexpected}")

    model.eval()
    model = model.to(device)
    if device.type == "cuda" and amp_dtype is not None:
        # Keep aggregator in AMP dtype during calibration for realistic activation range.
        model.aggregator.to(dtype=amp_dtype)

    def run_calibration() -> None:
        with torch.no_grad():
            total_batches = sum((len(paths) + args.batch_size - 1) // args.batch_size for paths in scene_to_image_paths.values())
            global_batch_idx = 0

            for scene_name, scene_paths in scene_to_image_paths.items():
                scene_total_batches = (len(scene_paths) + args.batch_size - 1) // args.batch_size
                for scene_batch_idx, image_batch in enumerate(batched(scene_paths, args.batch_size), start=1):
                    image_batch_str = [str(p) for p in image_batch]
                    images = load_and_preprocess_images(image_batch_str, mode=args.preprocess_mode).to(device)

                    amp_ctx = (
                        torch.cuda.amp.autocast(dtype=amp_dtype)
                        if (device.type == "cuda" and amp_dtype is not None)
                        else nullcontext()
                    )
                    with amp_ctx:
                        # Each forward only uses images from one scene.
                        _ = model(images)

                    global_batch_idx += 1
                    print(
                        f"Calibrated scene {scene_name} batch {scene_batch_idx}/{scene_total_batches} "
                        f"(global {global_batch_idx}/{total_batches})"
                    )

    artifact = calibrate_attention_scales(
        model=model,
        run_calibration=run_calibration,
        weight_qmax=args.weight_qmax,
        act_qmax=args.act_qmax,
    )

    artifact["meta"].update(
        {
            "model_path": str(args.model_path),
            "calibration_source": source_type,
            "num_samples": total_images,
            "batch_size": int(args.batch_size),
            "preprocess_mode": args.preprocess_mode,
            "num_scenes": len(scene_to_image_paths),
        }
    )

    if source_type == "dtu":
        artifact["meta"].update(
            {
                "dtu_test_1200_path": str(args.dtu_test_1200_path),
                "dtu_scans": args.dtu_scans,
                "dtu_images_per_scene": int(args.dtu_images_per_scene),
            }
        )
    else:
        artifact["meta"].update({"calib_dir": str(args.calib_dir)})

    save_smoothquant_artifact(artifact, args.output_path)

    print(f"Saved SmoothQuant artifact to {args.output_path}")
    print(f"Calibrated layers: {artifact['meta']['num_layers']}")

    preview = sorted(artifact["stats"].items())[:5]
    for layer_name, layer_stats in preview:
        print(
            f"  {layer_name}: weight_max={layer_stats['weight_max']:.6g}, "
            f"act_max={layer_stats['act_max']:.6g}, scale={layer_stats['scale']:.6g}"
        )


if __name__ == "__main__":
    main()
