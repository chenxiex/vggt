import argparse
import random
from contextlib import nullcontext
import os
from pathlib import Path
from typing import Iterable, List
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
    parser.add_argument("--calib_dir", type=Path, required=True, help="Calibration image directory")
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

    image_paths = collect_image_paths(args.calib_dir, recursive=args.recursive)
    if not image_paths:
        raise ValueError(f"No images found in calibration directory: {args.calib_dir}")

    image_paths = sample_image_paths(image_paths, args.num_samples, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if device.type == "cuda" and not args.disable_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print(f"Calibration images: {len(image_paths)}")
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
            total_batches = (len(image_paths) + args.batch_size - 1) // args.batch_size
            for batch_idx, image_batch in enumerate(batched(image_paths, args.batch_size), start=1):
                image_batch_str = [str(p) for p in image_batch]
                images = load_and_preprocess_images(image_batch_str, mode=args.preprocess_mode).to(device)

                amp_ctx = (
                    torch.amp.autocast("cuda", dtype=amp_dtype)
                    if (device.type == "cuda" and amp_dtype is not None)
                    else nullcontext()
                )
                with amp_ctx:
                    _ = model(images)

                print(f"Calibrated batch {batch_idx}/{total_batches}")

    artifact = calibrate_attention_scales(
        model=model,
        run_calibration=run_calibration,
        weight_qmax=args.weight_qmax,
        act_qmax=args.act_qmax,
    )

    artifact["meta"].update(
        {
            "model_path": str(args.model_path),
            "calib_dir": str(args.calib_dir),
            "num_samples": len(image_paths),
            "batch_size": int(args.batch_size),
            "preprocess_mode": args.preprocess_mode,
        }
    )

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
