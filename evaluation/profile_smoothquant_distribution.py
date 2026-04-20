import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.request import urlretrieve

import torch

from vggt.models.vggt import VGGT
from vggt.quantization.smoothquant import find_attention_linear_layers
from vggt.utils.load_fn import load_and_preprocess_images

HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co")
MODEL_URL = f"{HF_ENDPOINT}/facebook/VGGT-1B/resolve/main/model.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class RunningDistributionStats:
    def __init__(self, hist_bins: int, log_abs_min: float, log_abs_max: float, zero_eps: float) -> None:
        if hist_bins <= 0:
            raise ValueError(f"hist_bins must be positive, got {hist_bins}")
        if log_abs_max <= log_abs_min:
            raise ValueError("hist_log_max must be larger than hist_log_min")
        if zero_eps <= 0.0:
            raise ValueError("zero_eps must be positive")

        self.hist_bins = int(hist_bins)
        self.log_abs_min = float(log_abs_min)
        self.log_abs_max = float(log_abs_max)
        self.zero_eps = float(zero_eps)

        self.log_edges = torch.linspace(self.log_abs_min, self.log_abs_max, self.hist_bins + 1, dtype=torch.float32)
        self.hist_counts = torch.zeros(self.hist_bins, dtype=torch.int64)

        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.abs_sum = 0.0
        self.min_value: Optional[float] = None
        self.max_value: Optional[float] = None
        self.abs_max = 0.0

        self.zero_count = 0
        self.positive_count = 0
        self.negative_count = 0

        self.underflow_count = 0
        self.overflow_count = 0

    def update(self, tensor: torch.Tensor) -> None:
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            return

        values = tensor.detach().float().reshape(-1).cpu()
        count = int(values.numel())
        if count == 0:
            return

        self.count += count
        self.sum += float(values.sum(dtype=torch.float64).item())
        self.sum_sq += float(torch.square(values).sum(dtype=torch.float64).item())

        current_min = float(values.min().item())
        current_max = float(values.max().item())
        self.min_value = current_min if self.min_value is None else min(self.min_value, current_min)
        self.max_value = current_max if self.max_value is None else max(self.max_value, current_max)

        abs_values = values.abs()
        self.abs_sum += float(abs_values.sum(dtype=torch.float64).item())
        self.abs_max = max(self.abs_max, float(abs_values.max().item()))

        self.zero_count += int((abs_values <= self.zero_eps).sum().item())
        self.positive_count += int((values > 0).sum().item())
        self.negative_count += int((values < 0).sum().item())

        log_abs = torch.log10(abs_values + self.zero_eps)
        bin_ids = torch.bucketize(log_abs, self.log_edges, right=False) - 1

        under_mask = bin_ids < 0
        over_mask = bin_ids >= self.hist_bins

        self.underflow_count += int(under_mask.sum().item())
        self.overflow_count += int(over_mask.sum().item())

        valid = bin_ids[(~under_mask) & (~over_mask)]
        if valid.numel() > 0:
            counts = torch.bincount(valid, minlength=self.hist_bins)
            self.hist_counts += counts.to(torch.int64)

    def _estimate_abs_percentile(self, percentile: float) -> float:
        if self.count <= 0:
            return 0.0

        if percentile <= 0.0:
            return 0.0

        if percentile >= 100.0:
            return float(self.abs_max)

        hist_total = int(self.hist_counts.sum().item())
        total = self.underflow_count + hist_total + self.overflow_count
        if total <= 0:
            return 0.0

        target = (percentile / 100.0) * (total - 1)
        cumulative = float(self.underflow_count)

        if target < cumulative:
            return 0.0

        for idx in range(self.hist_bins):
            bin_count = int(self.hist_counts[idx].item())
            if bin_count <= 0:
                continue

            next_cumulative = cumulative + bin_count
            if target < next_cumulative:
                left = float(self.log_edges[idx].item())
                right = float(self.log_edges[idx + 1].item())
                frac = (target - cumulative) / max(bin_count, 1)
                frac = min(max(frac, 0.0), 1.0)
                log_value = left + frac * (right - left)
                value = (10.0 ** log_value) - self.zero_eps
                return float(max(value, 0.0))
            cumulative = next_cumulative

        return float(self.abs_max)

    @staticmethod
    def _percentile_key(percentile: float) -> str:
        if float(percentile).is_integer():
            return str(int(percentile))
        text = f"{percentile}".rstrip("0").rstrip(".")
        return text

    def to_dict(self, percentiles: Sequence[float]) -> Dict[str, Any]:
        if self.count <= 0:
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "abs_mean": 0.0,
                "abs_max": 0.0,
                "zero_ratio": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "abs_percentiles": {self._percentile_key(p): 0.0 for p in percentiles},
                "abs_log10_histogram": {
                    "bin_edges": [float(v) for v in self.log_edges.tolist()],
                    "counts": [int(v) for v in self.hist_counts.tolist()],
                    "underflow_count": int(self.underflow_count),
                    "overflow_count": int(self.overflow_count),
                },
            }

        mean = self.sum / self.count
        variance = max(self.sum_sq / self.count - mean * mean, 0.0)

        return {
            "count": int(self.count),
            "min": float(self.min_value if self.min_value is not None else 0.0),
            "max": float(self.max_value if self.max_value is not None else 0.0),
            "mean": float(mean),
            "std": float(math.sqrt(variance)),
            "abs_mean": float(self.abs_sum / self.count),
            "abs_max": float(self.abs_max),
            "zero_ratio": float(self.zero_count / self.count),
            "positive_ratio": float(self.positive_count / self.count),
            "negative_ratio": float(self.negative_count / self.count),
            "abs_percentiles": {
                self._percentile_key(p): float(self._estimate_abs_percentile(p)) for p in percentiles
            },
            "abs_log10_histogram": {
                "bin_edges": [float(v) for v in self.log_edges.tolist()],
                "counts": [int(v) for v in self.hist_counts.tolist()],
                "underflow_count": int(self.underflow_count),
                "overflow_count": int(self.overflow_count),
            },
        }


def parse_percentiles(text: str) -> List[float]:
    values: List[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value <= 0.0 or value > 100.0:
            raise ValueError(f"Percentile should be in (0, 100], got {value}")
        values.append(value)

    if not values:
        raise ValueError("No valid percentiles provided")

    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile activation and weight distributions for VGGT SmoothQuant attention layers"
    )
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
    parser.add_argument("--output_path", type=Path, required=True, help="Output path for statistics (.json or .pt)")
    parser.add_argument("--num_samples", type=int, default=32, help="Maximum number of calibration images")
    parser.add_argument("--batch_size", type=int, default=8, help="Calibration batch size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for calibration sampling")
    parser.add_argument("--preprocess_mode", type=str, choices=["crop", "pad"], default="crop")
    parser.add_argument("--recursive", action="store_true", help="Recursively search images under calib_dir")
    parser.add_argument("--disable_amp", action="store_true", help="Disable AMP during profiling")

    parser.add_argument("--hist_bins", type=int, default=256, help="Number of bins for log10(abs(x)) histogram")
    parser.add_argument("--hist_log_min", type=float, default=-8.0, help="Lower bound of log10(abs(x)) histogram")
    parser.add_argument("--hist_log_max", type=float, default=2.0, help="Upper bound of log10(abs(x)) histogram")
    parser.add_argument("--zero_eps", type=float, default=1e-8, help="Epsilon used for abs(x) stability and near-zero count")
    parser.add_argument(
        "--percentiles",
        type=str,
        default="50,90,95,99,99.9",
        help="Comma-separated absolute-value percentiles to report",
    )
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
        raise ValueError("No DTU scenes found for profiling")

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

        scene_to_paths = {scene_name: paths for scene_name, paths in truncated.items() if paths}

    return scene_to_paths


def collect_image_paths(calib_dir: Path, recursive: bool) -> List[Path]:
    if not calib_dir.exists():
        raise FileNotFoundError(f"Calibration directory not found: {calib_dir}")

    image_root = calib_dir / "images" if (calib_dir / "images").is_dir() else calib_dir
    entries = image_root.rglob("*") if recursive else image_root.glob("*")

    image_paths = [p for p in entries if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(image_paths)


def sample_image_paths(image_paths: List[Path], num_samples: int, seed: int) -> List[Path]:
    if num_samples <= 0 or len(image_paths) <= num_samples:
        return image_paths
    rng = random.Random(seed)
    return sorted(rng.sample(image_paths, num_samples))


def batched(items: List[Path], batch_size: int) -> Iterable[List[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_state_dict(model_path: Path) -> Dict[str, Any]:
    if not model_path.exists():
        print(f"Checkpoint not found at {model_path}, downloading from {MODEL_URL}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(MODEL_URL, model_path)

    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def create_stats_collector(args: argparse.Namespace) -> RunningDistributionStats:
    return RunningDistributionStats(
        hist_bins=args.hist_bins,
        log_abs_min=args.hist_log_min,
        log_abs_max=args.hist_log_max,
        zero_eps=args.zero_eps,
    )


def safe_divide(numerator: float, denominator: float) -> Optional[float]:
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def percentile_key(percentile: float) -> str:
    if float(percentile).is_integer():
        return str(int(percentile))
    return f"{percentile}".rstrip("0").rstrip(".")


def build_comparison(
    weight_layer_stats: Dict[str, Dict[str, Any]],
    act_layer_stats: Dict[str, Dict[str, Any]],
    weight_global_stats: Dict[str, Any],
    act_global_stats: Dict[str, Any],
    percentiles: Sequence[float],
) -> Dict[str, Any]:
    percentile_keys = [percentile_key(p) for p in percentiles]

    global_comp: Dict[str, Optional[float]] = {
        "abs_max_ratio_act_over_weight": safe_divide(act_global_stats["abs_max"], weight_global_stats["abs_max"]),
    }
    for key in percentile_keys:
        act_p = act_global_stats["abs_percentiles"][key]
        weight_p = weight_global_stats["abs_percentiles"][key]
        global_comp[f"abs_p{key}_ratio_act_over_weight"] = safe_divide(act_p, weight_p)

    per_layer_comp: Dict[str, Dict[str, Optional[float]]] = {}
    for layer_name in sorted(act_layer_stats):
        if layer_name not in weight_layer_stats:
            continue
        act = act_layer_stats[layer_name]
        weight = weight_layer_stats[layer_name]

        comp: Dict[str, Optional[float]] = {
            "abs_max_ratio_act_over_weight": safe_divide(act["abs_max"], weight["abs_max"])
        }
        for key in percentile_keys:
            act_p = act["abs_percentiles"][key]
            weight_p = weight["abs_percentiles"][key]
            comp[f"abs_p{key}_ratio_act_over_weight"] = safe_divide(act_p, weight_p)

        per_layer_comp[layer_name] = comp

    return {
        "global": global_comp,
        "per_layer": per_layer_comp,
    }


def save_results(results: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".pt":
        torch.save(results, output_path)
        return

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if args.num_samples == 0:
        raise ValueError("num_samples cannot be 0; use a positive value or -1 for all")

    if args.dtu_images_per_scene < 0:
        raise ValueError("dtu_images_per_scene must be >= 0")

    if args.calib_dir is None and args.dtu_test_1200_path is None:
        raise ValueError("Either --calib_dir or --dtu_test_1200_path must be provided")

    if args.calib_dir is not None and args.dtu_test_1200_path is not None:
        raise ValueError("--calib_dir and --dtu_test_1200_path are mutually exclusive")

    percentiles = parse_percentiles(args.percentiles)

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

    print(f"Profiling source: {source_type}")
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
        model.aggregator.to(dtype=amp_dtype)

    layers = find_attention_linear_layers(model)
    if not layers:
        raise RuntimeError("No attention qkv/proj linear layers found in model")

    weight_collectors = {layer_name: create_stats_collector(args) for layer_name in layers}
    weight_global_collector = create_stats_collector(args)

    for layer_name, linear in layers.items():
        weight_collectors[layer_name].update(linear.weight)
        weight_global_collector.update(linear.weight)

    activation_collectors = {layer_name: create_stats_collector(args) for layer_name in layers}
    activation_global_collector = create_stats_collector(args)
    activation_update_counts = {layer_name: 0 for layer_name in layers}

    handles = []
    for layer_name, linear in layers.items():

        def _pre_hook(module: torch.nn.Module, inputs: tuple[Any, ...], name: str = layer_name) -> None:
            del module
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            activation_collectors[name].update(x)
            activation_global_collector.update(x)
            activation_update_counts[name] += 1

        handles.append(linear.register_forward_pre_hook(_pre_hook))

    try:
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
                        _ = model(images)

                    global_batch_idx += 1
                    print(
                        f"Profiled scene {scene_name} batch {scene_batch_idx}/{scene_total_batches} "
                        f"(global {global_batch_idx}/{total_batches})"
                    )
    finally:
        for handle in handles:
            handle.remove()

    weight_layer_stats = {
        layer_name: collector.to_dict(percentiles) for layer_name, collector in sorted(weight_collectors.items())
    }
    weight_global_stats = weight_global_collector.to_dict(percentiles)

    activation_layer_stats = {
        layer_name: collector.to_dict(percentiles) for layer_name, collector in sorted(activation_collectors.items())
    }
    activation_global_stats = activation_global_collector.to_dict(percentiles)

    comparison = build_comparison(
        weight_layer_stats=weight_layer_stats,
        act_layer_stats=activation_layer_stats,
        weight_global_stats=weight_global_stats,
        act_global_stats=activation_global_stats,
        percentiles=percentiles,
    )

    results: Dict[str, Any] = {
        "meta": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_path": str(args.model_path),
            "calibration_source": source_type,
            "num_samples": int(total_images),
            "batch_size": int(args.batch_size),
            "preprocess_mode": args.preprocess_mode,
            "num_scenes": int(len(scene_to_image_paths)),
            "num_layers": int(len(layers)),
            "hist_bins": int(args.hist_bins),
            "hist_log_min": float(args.hist_log_min),
            "hist_log_max": float(args.hist_log_max),
            "zero_eps": float(args.zero_eps),
            "percentiles": [float(v) for v in percentiles],
            "device": str(device),
            "amp_dtype": str(amp_dtype) if amp_dtype is not None else None,
        },
        "weights": {
            "global": weight_global_stats,
            "per_layer": weight_layer_stats,
        },
        "activations": {
            "global": activation_global_stats,
            "per_layer": activation_layer_stats,
            "update_counts": activation_update_counts,
        },
        "comparison": comparison,
    }

    if source_type == "dtu":
        results["meta"].update(
            {
                "dtu_test_1200_path": str(args.dtu_test_1200_path),
                "dtu_scans": args.dtu_scans,
                "dtu_images_per_scene": int(args.dtu_images_per_scene),
            }
        )
    else:
        results["meta"].update({"calib_dir": str(args.calib_dir)})

    save_results(results, args.output_path)

    print(f"Saved distribution stats to {args.output_path}")
    print(
        "Global abs-max ratio (act/weight): "
        f"{comparison['global']['abs_max_ratio_act_over_weight']}"
    )

    top_by_ratio = []
    for layer_name, layer_comp in comparison["per_layer"].items():
        ratio = layer_comp.get("abs_max_ratio_act_over_weight")
        if ratio is None:
            continue
        top_by_ratio.append((float(ratio), layer_name))

    top_by_ratio.sort(reverse=True)
    preview = top_by_ratio[:5]
    if preview:
        print("Top-5 layers by abs-max ratio (act/weight):")
        for ratio, layer_name in preview:
            print(f"  {layer_name}: {ratio:.6g}")


if __name__ == "__main__":
    main()
