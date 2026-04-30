#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from utils import load_model, predict, write_ply
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_IMAGE_SUBDIR = Path("images") / "dslr_images_undistorted"
DEFAULT_SPARSE_SUBDIR = Path("dslr_calibration_undistorted")


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _torch_load_cpu_maybe_mmap(file_path: Path):
    try:
        return torch.load(file_path, map_location=torch.device("cpu"), mmap=True)
    except TypeError:
        return torch.load(file_path, map_location=torch.device("cpu"))
    except RuntimeError as exc:
        if "mmap" in str(exc).lower():
            return torch.load(file_path, map_location=torch.device("cpu"))
        raise


def save_predictions(prediction_file: Path, predictions: dict, image_relpaths: list[str]) -> None:
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "predictions": {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in predictions.items()
            if key in {"images", "pose_enc", "world_points", "world_points_conf"}
        },
        "image_relpaths": image_relpaths,
    }
    torch.save(payload, prediction_file)


def load_predictions(prediction_file: Path) -> tuple[dict, list[str]]:
    payload = _torch_load_cpu_maybe_mmap(prediction_file)
    predictions = payload["predictions"]
    image_relpaths = payload["image_relpaths"]

    required_keys = {"images", "pose_enc", "world_points", "world_points_conf"}
    missing = sorted(required_keys - set(predictions.keys()))
    if missing:
        raise KeyError(f"Prediction file {prediction_file} is missing keys: {missing}")

    return predictions, list(image_relpaths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGGT on ETH3D undistorted scenes and export aligned point clouds."
    )
    parser.add_argument(
        "--eth3d_root",
        type=Path,
        required=True,
        help="ETH3D undistorted root, e.g. data/ETH3D/train/undistorted",
    )
    parser.add_argument(
        "--results_path",
        type=Path,
        required=True,
        help="Directory where per-scene outputs will be written.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=False,
        help="VGGT checkpoint path. Required unless --no_pred is set.",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="all",
        help="Comma-separated scene names to process, or 'all'.",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=3.0,
        help="Keep only points with world_points_conf >= this value.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=300000,
        help="Maximum number of points kept per scene after confidence filtering. <=0 keeps all.",
    )
    parser.add_argument(
        "--no_pred",
        action="store_true",
        help="Reuse existing predictions.pt files instead of running inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for deterministic point subsampling and robust alignment.",
    )
    parser.add_argument(
        "--ransac_iters",
        type=int,
        default=256,
        help="RANSAC iterations for camera-center similarity alignment.",
    )
    parser.add_argument(
        "--ransac_threshold",
        type=float,
        default=0.25,
        help="Inlier threshold in world units for camera-center alignment residuals.",
    )
    return parser.parse_args()


def discover_scene_dirs(eth3d_root: Path, scenes_arg: str) -> list[Path]:
    if not eth3d_root.exists():
        raise FileNotFoundError(f"ETH3D root does not exist: {eth3d_root}")

    all_scenes = sorted(
        scene_dir for scene_dir in eth3d_root.iterdir()
        if scene_dir.is_dir()
        and (scene_dir / DEFAULT_IMAGE_SUBDIR).is_dir()
        and (scene_dir / DEFAULT_SPARSE_SUBDIR).is_dir()
    )
    if scenes_arg.lower() == "all":
        return all_scenes

    requested = {token.strip() for token in scenes_arg.split(",") if token.strip()}
    selected = [scene_dir for scene_dir in all_scenes if scene_dir.name in requested]
    missing = sorted(requested - {scene_dir.name for scene_dir in selected})
    for scene_name in missing:
        logger.warning("Scene '%s' was requested but not found under %s", scene_name, eth3d_root)
    return selected


def collect_image_paths(scene_dir: Path) -> tuple[list[Path], list[str]]:
    images_root = scene_dir / "images"
    if not images_root.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_root}")

    image_paths = sorted(
        path for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found under {images_root}")

    image_relpaths = [path.relative_to(images_root).as_posix() for path in image_paths]
    return image_paths, image_relpaths


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    norm = np.linalg.norm(qvec)
    if norm == 0:
        raise ValueError("Quaternion has zero norm.")
    w, x, y, z = qvec / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def parse_colmap_images(images_txt: Path) -> dict[str, np.ndarray]:
    if not images_txt.exists():
        raise FileNotFoundError(f"Missing COLMAP images file: {images_txt}")

    image_centers: dict[str, np.ndarray] = {}
    with images_txt.open("r", encoding="utf-8") as fp:
        lines = [line.rstrip("\n") for line in fp]

    data_lines = [line for line in lines if line and not line.startswith("#")]
    if len(data_lines) % 2 != 0:
        raise ValueError(f"Unexpected COLMAP images.txt format in {images_txt}")

    for idx in range(0, len(data_lines), 2):
        tokens = data_lines[idx].split()
        if len(tokens) < 10:
            raise ValueError(f"Malformed images.txt line: {data_lines[idx]}")

        qvec = np.array([float(v) for v in tokens[1:5]], dtype=np.float64)
        tvec = np.array([float(v) for v in tokens[5:8]], dtype=np.float64)
        rel_name = tokens[9]

        rot = qvec_to_rotmat(qvec)
        center = -(rot.T @ tvec)
        image_centers[rel_name.lower()] = center.astype(np.float64)

    return image_centers


def parse_points3d_count(points3d_txt: Path) -> int:
    if not points3d_txt.exists():
        raise FileNotFoundError(f"Missing COLMAP points3D file: {points3d_txt}")

    count = 0
    with points3d_txt.open("r", encoding="utf-8") as fp:
        for line in fp:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
    return count


def parse_cameras_count(cameras_txt: Path) -> int:
    if not cameras_txt.exists():
        raise FileNotFoundError(f"Missing COLMAP cameras file: {cameras_txt}")

    count = 0
    with cameras_txt.open("r", encoding="utf-8") as fp:
        for line in fp:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
    return count


def decode_pred_extrinsics(predictions: dict) -> np.ndarray:
    image_hw = tuple(predictions["images"].shape[-2:])
    extrinsic, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], image_hw)
    return extrinsic[0].detach().cpu().numpy().astype(np.float64)


def extrinsics_to_centers(extrinsics: np.ndarray) -> np.ndarray:
    rot = extrinsics[:, :, :3]
    trans = extrinsics[:, :, 3]
    return (-np.einsum("nij,nj->ni", np.transpose(rot, (0, 2, 1)), trans)).astype(np.float64)


def build_gt_matches(image_relpaths: list[str], gt_centers_by_name: dict[str, np.ndarray]) -> tuple[list[int], np.ndarray]:
    matched_indices: list[int] = []
    matched_gt_centers: list[np.ndarray] = []

    basename_to_centers: dict[str, np.ndarray] = {}
    for rel_name, center in gt_centers_by_name.items():
        basename = Path(rel_name).name.lower()
        basename_to_centers[basename] = center

    for idx, rel_name in enumerate(image_relpaths):
        key = rel_name.lower()
        gt_center = gt_centers_by_name.get(key)
        if gt_center is None:
            gt_center = basename_to_centers.get(Path(key).name.lower())
        if gt_center is None:
            logger.warning("No GT camera found for image '%s', skipping it for alignment.", rel_name)
            continue
        matched_indices.append(idx)
        matched_gt_centers.append(gt_center)

    if not matched_indices:
        raise RuntimeError("No overlapping image names were found between predictions and COLMAP images.txt")

    return matched_indices, np.stack(matched_gt_centers, axis=0)


def fit_similarity_umeyama(src_points: np.ndarray, dst_points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if src_points.shape != dst_points.shape or src_points.ndim != 2 or src_points.shape[1] != 3:
        raise ValueError("src_points and dst_points must both have shape Nx3")
    if src_points.shape[0] < 3:
        raise ValueError("At least 3 point correspondences are required for similarity alignment")

    src_mean = src_points.mean(axis=0)
    dst_mean = dst_points.mean(axis=0)
    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean

    cov = (dst_centered.T @ src_centered) / src_points.shape[0]
    u, singular_values, vh = np.linalg.svd(cov)
    d = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vh) < 0:
        d[-1, -1] = -1.0
    rot = u @ d @ vh

    src_var = np.mean(np.sum(src_centered ** 2, axis=1))
    if src_var <= 1e-12:
        raise ValueError("Source points are degenerate; cannot estimate similarity transform")

    scale = float(np.sum(singular_values * np.diag(d)) / src_var)
    trans = dst_mean - scale * (rot @ src_mean)
    return scale, rot, trans


def apply_similarity(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return (scale * (points @ rot.T)) + trans


def compute_residuals(src_points: np.ndarray, dst_points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    aligned = apply_similarity(src_points, scale, rot, trans)
    return np.linalg.norm(aligned - dst_points, axis=1)


def robust_align_cameras(
    pred_centers: np.ndarray,
    gt_centers: np.ndarray,
    seed: int,
    ransac_iters: int,
    ransac_threshold: float,
) -> dict:
    num_points = pred_centers.shape[0]
    if num_points < 3:
        raise RuntimeError(f"Need at least 3 matched cameras for alignment, got {num_points}")

    if num_points == 3:
        scale, rot, trans = fit_similarity_umeyama(pred_centers, gt_centers)
        residuals = compute_residuals(pred_centers, gt_centers, scale, rot, trans)
        inlier_mask = np.ones((num_points,), dtype=bool)
    else:
        rng = random.Random(seed)
        best_inlier_mask: Optional[np.ndarray] = None
        best_inlier_count = -1
        best_residual_median = math.inf

        indices = list(range(num_points))
        for _ in range(max(1, ransac_iters)):
            sample_ids = rng.sample(indices, 3)
            try:
                scale_i, rot_i, trans_i = fit_similarity_umeyama(
                    pred_centers[sample_ids], gt_centers[sample_ids]
                )
            except ValueError:
                continue

            residuals_i = compute_residuals(pred_centers, gt_centers, scale_i, rot_i, trans_i)
            inlier_mask_i = residuals_i <= ransac_threshold
            inlier_count = int(inlier_mask_i.sum())
            residual_median = float(np.median(residuals_i[inlier_mask_i])) if inlier_count > 0 else math.inf

            if (
                inlier_count > best_inlier_count
                or (inlier_count == best_inlier_count and residual_median < best_residual_median)
            ):
                best_inlier_mask = inlier_mask_i
                best_inlier_count = inlier_count
                best_residual_median = residual_median

        if best_inlier_mask is None or best_inlier_count < 3:
            raise RuntimeError("RANSAC failed to find a valid camera alignment with at least 3 inliers")

        scale, rot, trans = fit_similarity_umeyama(
            pred_centers[best_inlier_mask], gt_centers[best_inlier_mask]
        )
        residuals = compute_residuals(pred_centers, gt_centers, scale, rot, trans)
        inlier_mask = residuals <= ransac_threshold
        if int(inlier_mask.sum()) >= 3:
            scale, rot, trans = fit_similarity_umeyama(
                pred_centers[inlier_mask], gt_centers[inlier_mask]
            )
            residuals = compute_residuals(pred_centers, gt_centers, scale, rot, trans)

    return {
        "scale": float(scale),
        "rotation": rot.astype(np.float64),
        "translation": trans.astype(np.float64),
        "residuals": residuals.astype(np.float64),
        "inlier_mask": inlier_mask.astype(bool),
    }


def build_point_cloud(
    predictions: dict,
    conf_threshold: float,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    world_points = predictions["world_points"][0].detach().cpu().numpy()
    world_points_conf = predictions["world_points_conf"][0].detach().cpu().numpy()
    images = predictions["images"][0].detach().cpu().numpy().transpose(0, 2, 3, 1)

    points = world_points.reshape(-1, 3)
    conf = world_points_conf.reshape(-1)
    colors = np.clip(np.rint(images.reshape(-1, 3) * 255.0), 0, 255).astype(np.uint8)

    finite_conf_mask = np.isfinite(conf)
    finite_conf = conf[finite_conf_mask]
    if finite_conf.size == 0:
        raise RuntimeError("world_points_conf does not contain any finite values")

    valid_mask = np.isfinite(points).all(axis=1) & finite_conf_mask & (conf >= conf_threshold)
    kept_indices = np.flatnonzero(valid_mask)
    if kept_indices.size == 0:
        raise RuntimeError(f"No points survived confidence filtering at threshold {conf_threshold}")

    if max_points > 0 and kept_indices.size > max_points:
        rng = np.random.default_rng(seed)
        kept_indices = np.sort(rng.choice(kept_indices, size=max_points, replace=False))

    point_cloud = np.concatenate([points[kept_indices], colors[kept_indices]], axis=1)
    stats = {
        "num_points_before_filtering": int(points.shape[0]),
        "num_points_after_filtering": int(valid_mask.sum()),
        "num_points_written": int(point_cloud.shape[0]),
        "confidence_threshold": float(conf_threshold),
        "confidence_min": float(np.min(finite_conf)),
        "confidence_max": float(np.max(finite_conf)),
        "confidence_median": float(np.median(finite_conf)),
        "confidence_mean": float(np.mean(finite_conf)),
        "confidence_std": float(np.std(finite_conf)),
        "num_finite_confidences": int(finite_conf.size),
        "num_confidences_above_threshold": int(np.count_nonzero(finite_conf >= conf_threshold)),
    }
    return point_cloud, stats


def write_json(file_path: Path, payload: dict) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def process_scene(args: argparse.Namespace, scene_dir: Path, model) -> None:
    scene_name = scene_dir.name
    logger.info("Processing ETH3D scene %s", scene_name)

    scene_output_dir = args.results_path / scene_name
    prediction_file = scene_output_dir / "predictions.pt"
    raw_ply_file = scene_output_dir / "raw_points.ply"
    aligned_ply_file = scene_output_dir / "aligned_points.ply"
    alignment_json_file = scene_output_dir / "alignment.json"
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    image_paths, image_relpaths = collect_image_paths(scene_dir)

    if args.no_pred:
        predictions, cached_relpaths = load_predictions(prediction_file)
        if [p.lower() for p in cached_relpaths] != [p.lower() for p in image_relpaths]:
            raise RuntimeError(
                f"Cached prediction image list for {scene_name} does not match current scene files. "
                "Re-run without --no_pred to refresh predictions."
            )
    else:
        if model is None:
            raise ValueError("Model must be loaded when running prediction.")
        logger.info("Running VGGT inference on %d images for scene %s", len(image_paths), scene_name)
        predictions = predict(image_paths, model)
        save_predictions(prediction_file, predictions, image_relpaths)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Building raw point cloud for %s", scene_name)
    point_cloud, point_stats = build_point_cloud(
        predictions,
        conf_threshold=args.conf_threshold,
        max_points=args.max_points,
        seed=args.seed + sum(ord(ch) for ch in scene_name),
    )
    logger.info(
        (
            "Scene %s points conf stats: min=%.4f, max=%.4f, median=%.4f, "
            "mean=%.4f, std=%.4f, finite=%d, kept>=%0.4f: %d/%d, written=%d"
        ),
        scene_name,
        point_stats["confidence_min"],
        point_stats["confidence_max"],
        point_stats["confidence_median"],
        point_stats["confidence_mean"],
        point_stats["confidence_std"],
        point_stats["num_finite_confidences"],
        point_stats["confidence_threshold"],
        point_stats["num_confidences_above_threshold"],
        point_stats["num_points_before_filtering"],
        point_stats["num_points_written"],
    )
    write_ply(raw_ply_file, point_cloud)

    sparse_dir = scene_dir / DEFAULT_SPARSE_SUBDIR
    gt_camera_count = parse_cameras_count(sparse_dir / "cameras.txt")
    gt_centers_by_name = parse_colmap_images(sparse_dir / "images.txt")
    gt_sparse_point_count = parse_points3d_count(sparse_dir / "points3D.txt")

    matched_indices, gt_centers = build_gt_matches(image_relpaths, gt_centers_by_name)
    pred_extrinsics = decode_pred_extrinsics(predictions)
    pred_centers_all = extrinsics_to_centers(pred_extrinsics)
    pred_centers = pred_centers_all[matched_indices]

    logger.info(
        "Aligning %s using %d matched cameras against COLMAP sparse model",
        scene_name,
        len(matched_indices),
    )
    alignment = robust_align_cameras(
        pred_centers,
        gt_centers,
        seed=args.seed + len(matched_indices),
        ransac_iters=args.ransac_iters,
        ransac_threshold=args.ransac_threshold,
    )

    aligned_xyz = apply_similarity(
        point_cloud[:, :3].astype(np.float64),
        alignment["scale"],
        alignment["rotation"],
        alignment["translation"],
    )
    aligned_point_cloud = np.concatenate([aligned_xyz, point_cloud[:, 3:]], axis=1)
    write_ply(aligned_ply_file, aligned_point_cloud)

    residuals = alignment["residuals"]
    inlier_mask = alignment["inlier_mask"]
    payload = {
        "scene_name": scene_name,
        "scene_dir": str(scene_dir),
        "prediction_file": str(prediction_file),
        "raw_ply": str(raw_ply_file),
        "aligned_ply": str(aligned_ply_file),
        "recommended_eval_ply": str(aligned_ply_file),
        "num_images": len(image_relpaths),
        "num_matched_cameras": int(len(matched_indices)),
        "num_alignment_inliers": int(inlier_mask.sum()),
        "gt_camera_count": gt_camera_count,
        "gt_sparse_point_count": gt_sparse_point_count,
        "point_stats": point_stats,
        "alignment": {
            "scale": alignment["scale"],
            "rotation": alignment["rotation"].tolist(),
            "translation": alignment["translation"].tolist(),
            "ransac_threshold": float(args.ransac_threshold),
            "residual_mean": float(np.mean(residuals)),
            "residual_median": float(np.median(residuals)),
            "residual_max": float(np.max(residuals)),
            "inlier_residual_mean": float(np.mean(residuals[inlier_mask])) if np.any(inlier_mask) else None,
            "inlier_residual_median": float(np.median(residuals[inlier_mask])) if np.any(inlier_mask) else None,
            "matched_image_relpaths": [image_relpaths[idx] for idx in matched_indices],
        },
    }
    write_json(alignment_json_file, payload)
    logger.info(
        "Scene %s finished. Use %s for downstream ETH3D evaluation.",
        scene_name,
        aligned_ply_file,
    )


def main() -> None:
    configure_logging()
    args = parse_args()

    if args.no_pred and args.model_path is not None:
        logger.info("--no_pred is set, ignoring --model_path=%s", args.model_path)
    if not args.no_pred and not args.model_path:
        raise ValueError("--model_path is required unless --no_pred is set.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scene_dirs = discover_scene_dirs(args.eth3d_root, args.scenes)
    if not scene_dirs:
        raise RuntimeError(f"No valid ETH3D scenes found under {args.eth3d_root}")

    model = None
    if not args.no_pred:
        logger.info("Loading VGGT model from %s", args.model_path)
        model = load_model(
            args.model_path,
            model_args={"enable_point": True, "enable_depth": False, "enable_track": False},
        )

    try:
        for scene_dir in scene_dirs:
            process_scene(args, scene_dir, model)
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
