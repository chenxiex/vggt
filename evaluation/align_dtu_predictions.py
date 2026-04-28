#!/usr/bin/env python3

"""Align DTU prediction files to ground-truth depths and average scale/shift."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from utils import read_pfm, upsample_image, upsample_images


logger = logging.getLogger(__name__)

MAX_IO_WORKERS = max(1, min(8, os.cpu_count() or 1))
MAX_UPSAMPLE_WORKERS = max(1, min(8, os.cpu_count() or 1))


def _torch_load_cpu_maybe_mmap(file_path: Path):
    try:
        return torch.load(file_path, map_location=torch.device("cpu"), mmap=True)
    except TypeError:
        return torch.load(file_path, map_location=torch.device("cpu"))
    except RuntimeError as exc:
        if "mmap" in str(exc).lower():
            return torch.load(file_path, map_location=torch.device("cpu"))
        raise


def configure_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


def load_predictions(results_path: Path, scene_name: str):
    results_file = results_path / f"{scene_name}.pt"
    results = _torch_load_cpu_maybe_mmap(results_file)
    sample_no = results["sample_no"]
    predictions = results["predictions"]

    if "depth" not in predictions or "depth_conf" not in predictions:
        raise KeyError(
            f"Prediction file {results_file} must contain 'depth' and 'depth_conf'."
        )

    predictions = {
        "depth": predictions["depth"],
        "depth_conf": predictions["depth_conf"],
    }
    return predictions, [int(i) for i in sample_no]


def load_gt_depth(gt_depths_path: Path, sample_no: list[int]):
    sampled_gt_depth_paths = [gt_depths_path / f"depth_map_{i:04}.pfm" for i in sample_no]

    if len(sampled_gt_depth_paths) <= 1:
        gt_depth = []
        for gt_depth_path in sampled_gt_depth_paths:
            data, scale = read_pfm(gt_depth_path)
            gt_depth.append(data * scale)
    else:
        def _read_scaled_depth(gt_depth_path: Path):
            data, scale = read_pfm(gt_depth_path)
            return data * scale

        with ThreadPoolExecutor(
            max_workers=min(MAX_IO_WORKERS, len(sampled_gt_depth_paths))
        ) as executor:
            gt_depth = list(executor.map(_read_scaled_depth, sampled_gt_depth_paths))

    gt_depth = torch.from_numpy(np.stack(gt_depth, axis=0)).float()
    return gt_depth


def align_pred_to_gt(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: np.ndarray,
    min_valid_pixels: int = 100,
):
    if pred_depth.shape != gt_depth.shape:
        raise ValueError(
            f"Predicted depth shape {pred_depth.shape} must match GT depth shape {gt_depth.shape}"
        )

    gt_masked = gt_depth[valid_mask]
    pred_masked = pred_depth[valid_mask]

    if len(gt_masked) < min_valid_pixels:
        logger.warning(
            "Not enough valid pixels (%d < %d) to align. Using all pixels.",
            len(gt_masked),
            min_valid_pixels,
        )
        gt_masked = gt_depth.reshape(-1)
        pred_masked = pred_depth.reshape(-1)

    if np.std(pred_masked) < 1e-6:
        logger.warning(
            "Predicted depth values in the valid mask have near-zero variance. "
            "Scale is ill-defined. Setting scale=1 and solving for shift only."
        )
        scale = 1.0
        shift = np.mean(gt_masked) - np.mean(pred_masked)
    else:
        A = np.vstack([pred_masked, np.ones_like(pred_masked)]).T
        try:
            x, _, _, _ = np.linalg.lstsq(A, gt_masked, rcond=None)
            scale, shift = x[0], x[1]
        except np.linalg.LinAlgError as exc:
            logger.warning(
                "Least squares alignment failed (%s). Returning invalid scale/shift.",
                exc,
            )
            return np.nan, np.nan

    return float(scale), float(shift)


def upsample_images_parallel(images: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
    num_images = int(images.shape[0])
    max_workers = min(MAX_UPSAMPLE_WORKERS, num_images)
    if max_workers <= 1:
        return upsample_images(images, target_w, target_h)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        upsampled_images = list(
            executor.map(lambda image: upsample_image(image, target_w, target_h), images)
        )

    return torch.stack(upsampled_images, dim=0)


def resolve_prediction_files(inputs: list[str]) -> list[Path]:
    prediction_files: list[Path] = []
    for item in inputs:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f"Prediction input does not exist: {path}")
        if path.is_dir():
            prediction_files.extend(sorted(path.glob("scan*.pt")))
        else:
            prediction_files.append(path)

    prediction_files = sorted({path.resolve() for path in prediction_files})
    if not prediction_files:
        raise ValueError("No prediction .pt files were found.")
    return prediction_files


def align_single_scan(
    prediction_file: Path,
    dtu_depths_path: Path,
    min_valid_pixels: int,
):
    scene_name = prediction_file.stem
    predictions, sample_no = load_predictions(prediction_file.parent, scene_name)

    gt_depths_path = dtu_depths_path / "Depths" / scene_name
    gt_depth = load_gt_depth(gt_depths_path, sample_no)
    gt_depth_w, gt_depth_h = gt_depth[0].shape[:2]

    depths = predictions["depth"][0]
    conf = predictions["depth_conf"][0]

    upsampled_pred_depth = upsample_images_parallel(depths, gt_depth_w, gt_depth_h)
    upsampled_depth_conf = upsample_images_parallel(conf, gt_depth_w, gt_depth_h)

    valid_mask = (gt_depth > 1e-3) & (upsampled_depth_conf > 3)

    scale_val, shift_val = align_pred_to_gt(
        upsampled_pred_depth.reshape(-1).cpu().numpy(),
        gt_depth.reshape(-1).cpu().numpy(),
        valid_mask.reshape(-1).cpu().numpy(),
        min_valid_pixels=min_valid_pixels,
    )

    valid_pixels = int(valid_mask.sum().item())
    return {
        "scene_name": scene_name,
        "prediction_file": str(prediction_file),
        "scale": scale_val,
        "shift": shift_val,
        "valid_pixels": valid_pixels,
        "sample_no": sample_no,
    }


def write_output(output_file: Path, payload: dict):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def main():
    configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prediction_inputs",
        nargs="+",
        help="Prediction .pt files or directories containing scan*.pt files.",
    )
    parser.add_argument(
        "--dtu_depths_path",
        type=Path,
        required=True,
        help="Path to the DTU raw depth maps.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path of the output JSON file.",
    )
    parser.add_argument(
        "--min_valid_pixels",
        type=int,
        default=100,
        help="Minimum valid pixels required for scale/shift fitting.",
    )
    args = parser.parse_args()

    prediction_files = resolve_prediction_files(args.prediction_inputs)
    logger.info("Aligning %d scan files.", len(prediction_files))

    per_scan_results = []
    for prediction_file in prediction_files:
        logger.info("Processing %s", prediction_file.name)
        per_scan_results.append(
            align_single_scan(prediction_file, args.dtu_depths_path, args.min_valid_pixels)
        )

    valid_results = [
        result for result in per_scan_results
        if np.isfinite(result["scale"]) and np.isfinite(result["shift"])
    ]
    if not valid_results:
        raise RuntimeError("No valid scan alignments were produced.")

    mean_scale = float(np.mean([result["scale"] for result in valid_results]))
    mean_shift = float(np.mean([result["shift"] for result in valid_results]))

    payload = {
        "mean_scale": mean_scale,
        "mean_shift": mean_shift,
        "num_scans": len(per_scan_results),
        "num_valid_scans": len(valid_results),
        "scans": per_scan_results,
    }
    write_output(args.output_file, payload)

    logger.info("Wrote averaged alignment to %s", args.output_file)
    logger.info("Mean scale: %.8f, mean shift: %.8f", mean_scale, mean_shift)


if __name__ == "__main__":
    main()