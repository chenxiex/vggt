import argparse
import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from utils import load_model, predict, write_ply
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export VGGT ETH3D point clouds for official evaluation")
    parser.add_argument(
        "--eth3d_root",
        type=Path,
        required=True,
        help="ETH3D root containing scene directories, e.g. data/ETH3D/train/undistorted",
    )
    parser.add_argument(
        "--results_path",
        type=Path,
        required=True,
        help="Directory to save per-scene metadata and PLY files",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to the VGGT checkpoint.",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default=None,
        help="Optional comma-separated list of scene names to process, e.g. courtyard,delivery_area",
    )
    parser.add_argument(
        "--point_conf_thresh",
        type=float,
        default=1.0,
        help="Confidence threshold for predicted world points.",
    )
    parser.add_argument(
        "--max_points_per_image",
        type=int,
        default=0,
        help="Optional cap on kept points per image after confidence filtering. <=0 disables sampling.",
    )
    parser.add_argument(
        "--alignment_mode",
        type=str,
        choices=["sim3", "rigid_first_frame", "none"],
        default="sim3",
        help="Alignment mode used to map predicted points to ETH3D world coordinates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic point sampling.",
    )
    parser.add_argument(
        "--smoothquant_scale_path",
        type=Path,
        default=None,
        help="Path to SmoothQuant calibration artifact (.pt) for W8A16 attention inference.",
    )
    parser.add_argument(
        "--smoothquant_allow_missing",
        action="store_true",
        help="Allow missing SmoothQuant scales for some attention layers.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="pseudo",
        choices=["pseudo", "torchao", "bitsandbytes", "trt_int8"],
        help="Quantization backend to use.",
    )
    return parser.parse_args()


def discover_scene_dirs(eth3d_root: Path, scenes_arg: Optional[str]) -> list[Path]:
    if not eth3d_root.exists():
        raise FileNotFoundError(f"ETH3D root path does not exist: {eth3d_root}")

    all_scene_dirs = sorted([path for path in eth3d_root.iterdir() if path.is_dir()])
    if scenes_arg is None or scenes_arg.strip() == "":
        return all_scene_dirs

    requested = [token.strip() for token in scenes_arg.split(",") if token.strip()]
    selected = []
    for scene_name in requested:
        scene_dir = eth3d_root / scene_name
        if scene_dir.is_dir():
            selected.append(scene_dir)
        else:
            logger.warning("Requested scene '%s' does not exist under %s, skipping.", scene_name, eth3d_root)
    return selected


def validate_scene_dir(scene_dir: Path) -> bool:
    required_paths = [
        scene_dir / "images",
        scene_dir / "dslr_calibration_undistorted" / "cameras.txt",
        scene_dir / "dslr_calibration_undistorted" / "images.txt",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        logger.warning("Skipping scene %s because required files are missing: %s", scene_dir.name, ", ".join(missing))
        return False
    return True


def parse_cameras_txt(cameras_txt: Path) -> dict[int, dict]:
    cameras: dict[int, dict] = {}
    with cameras_txt.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Malformed cameras.txt line in {cameras_txt}: {line}")

            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array([float(value) for value in parts[4:]], dtype=np.float64)
            cameras[camera_id] = {
                "camera_id": camera_id,
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    if not cameras:
        raise ValueError(f"No camera entries found in {cameras_txt}")
    return cameras


def quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        raise ValueError("Quaternion norm is zero.")
    qw, qx, qy, qz = quat / norm

    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def parse_images_txt(images_txt: Path) -> list[dict]:
    lines = images_txt.read_text(encoding="utf-8").splitlines()
    records: list[dict] = []

    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 10:
            raise ValueError(f"Malformed images.txt header line in {images_txt}: {line}")

        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = parts[9]

        rot = quat_to_rotmat(qw, qx, qy, qz)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = rot
        w2c[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)

        if idx < len(lines):
            idx += 1

        records.append(
            {
                "image_id": image_id,
                "camera_id": camera_id,
                "name": name,
                "w2c": w2c,
            }
        )

    if not records:
        raise ValueError(f"No image entries found in {images_txt}")
    return records


def collect_scene_records(scene_dir: Path) -> tuple[list[dict], dict]:
    calib_dir = scene_dir / "dslr_calibration_undistorted"
    cameras = parse_cameras_txt(calib_dir / "cameras.txt")
    image_records = parse_images_txt(calib_dir / "images.txt")

    usable_records = []
    missing_count = 0
    images_root = scene_dir / "images"
    image_candidates = list(images_root.rglob("*"))
    image_lookup: dict[str, Path] = {}
    for candidate in image_candidates:
        if not candidate.is_file():
            continue
        rel_path = candidate.relative_to(images_root)
        image_lookup[str(rel_path)] = candidate
        image_lookup[str(rel_path).lower()] = candidate
        image_lookup[rel_path.stem.lower()] = candidate

    for record in image_records:
        image_path = images_root / record["name"]
        if not image_path.exists():
            image_path = image_lookup.get(record["name"])
        if image_path is None or not image_path.exists():
            image_path = image_lookup.get(record["name"].lower())
        if image_path is None or not image_path.exists():
            image_path = image_lookup.get(Path(record["name"]).stem.lower())
        if image_path is None or not image_path.exists():
            logger.warning(
                "Scene %s is missing image referenced by images.txt: %s",
                scene_dir.name,
                record["name"],
            )
            missing_count += 1
            continue
        if record["camera_id"] not in cameras:
            logger.warning(
                "Scene %s references missing camera_id=%d for image %s",
                scene_dir.name,
                record["camera_id"],
                record["name"],
            )
            missing_count += 1
            continue
        merged = dict(record)
        merged["image_path"] = image_path
        merged["camera"] = cameras[record["camera_id"]]
        usable_records.append(merged)

    scene_info = {
        "scene_name": scene_dir.name,
        "total_images_in_txt": len(image_records),
        "missing_images": missing_count,
        "usable_images": len(usable_records),
    }
    return usable_records, scene_info


def decode_prediction_extrinsics(predictions: dict) -> torch.Tensor:
    if "pose_enc" not in predictions:
        raise KeyError("Prediction payload must contain 'pose_enc'.")
    image_hw = tuple(predictions["images_shape_hw"])
    pose_enc = predictions["pose_enc"]
    if not isinstance(pose_enc, torch.Tensor):
        pose_enc = torch.as_tensor(pose_enc)
    pred_extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, image_hw)
    pred_extrinsic = pred_extrinsic[0].float()

    num_frames = pred_extrinsic.shape[0]
    pred_w2c = torch.eye(4, dtype=torch.float32, device=pred_extrinsic.device).unsqueeze(0).repeat(num_frames, 1, 1)
    pred_w2c[:, :3, :4] = pred_extrinsic
    return pred_w2c


def camera_centers_from_w2c(w2c: np.ndarray) -> np.ndarray:
    rot = w2c[:, :3, :3]
    trans = w2c[:, :3, 3]
    centers = -np.einsum("nij,nj->ni", np.transpose(rot, (0, 2, 1)), trans)
    return centers


def umeyama_similarity(src: np.ndarray, dst: np.ndarray, estimate_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    if src.shape != dst.shape:
        raise ValueError(f"Source and destination must have the same shape, got {src.shape} vs {dst.shape}")
    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected Nx3 point sets, got {src.shape}")
    if src.shape[0] < 3:
        raise ValueError("At least 3 points are required for Sim(3) estimation.")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    cov = (dst_centered.T @ src_centered) / src.shape[0]
    u, singular_values, vh = np.linalg.svd(cov)
    rank = np.linalg.matrix_rank(cov)
    if rank < 2:
        raise ValueError("Point configuration is degenerate for Umeyama alignment.")

    d = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        d[-1, -1] = -1.0

    rotation = u @ d @ vh
    if estimate_scale:
        src_var = np.mean(np.sum(src_centered * src_centered, axis=1))
        if src_var < 1e-12:
            raise ValueError("Source points have near-zero variance; scale is ill-defined.")
        scale = float(np.sum(singular_values * np.diag(d)) / src_var)
    else:
        scale = 1.0

    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (points @ rotation.T) + translation[None, :]


def rigid_transform_from_first_frame(pred_w2c: np.ndarray, gt_w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_c2w = np.linalg.inv(pred_w2c)
    gt_c2w = np.linalg.inv(gt_w2c)
    transform = gt_c2w @ pred_w2c
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation


def align_points_to_eth3d(
    world_points: np.ndarray,
    pred_w2c: np.ndarray,
    gt_w2c: np.ndarray,
    alignment_mode: str,
) -> tuple[np.ndarray, dict]:
    pred_centers = camera_centers_from_w2c(pred_w2c)
    gt_centers = camera_centers_from_w2c(gt_w2c)
    stats: dict[str, object] = {
        "requested_alignment_mode": alignment_mode,
        "used_alignment_mode": alignment_mode,
        "sim3_success": False,
    }

    if alignment_mode == "none":
        before_error = np.linalg.norm(pred_centers - gt_centers, axis=1)
        stats["camera_center_error_before"] = {
            "mean": float(before_error.mean()),
            "median": float(np.median(before_error)),
            "max": float(before_error.max()),
        }
        return world_points, stats

    if alignment_mode == "sim3":
        try:
            scale, rotation, translation = umeyama_similarity(pred_centers, gt_centers, estimate_scale=True)
            aligned_points = apply_similarity(world_points, scale, rotation, translation)
            aligned_centers = apply_similarity(pred_centers, scale, rotation, translation)
            stats["sim3_success"] = True
            stats["sim3_scale"] = float(scale)
            stats["sim3_rotation"] = rotation.tolist()
            stats["sim3_translation"] = translation.tolist()
            before_error = np.linalg.norm(pred_centers - gt_centers, axis=1)
            after_error = np.linalg.norm(aligned_centers - gt_centers, axis=1)
            stats["camera_center_error_before"] = {
                "mean": float(before_error.mean()),
                "median": float(np.median(before_error)),
                "max": float(before_error.max()),
            }
            stats["camera_center_error_after"] = {
                "mean": float(after_error.mean()),
                "median": float(np.median(after_error)),
                "max": float(after_error.max()),
            }
            return aligned_points, stats
        except Exception as exc:
            logger.warning("Sim3 alignment failed: %s. Falling back to rigid_first_frame.", exc)
            stats["used_alignment_mode"] = "rigid_first_frame"
            stats["sim3_failure_reason"] = str(exc)
            alignment_mode = "rigid_first_frame"

    rotation, translation = rigid_transform_from_first_frame(pred_w2c[0], gt_w2c[0])
    aligned_points = world_points @ rotation.T + translation[None, :]
    aligned_centers = pred_centers @ rotation.T + translation[None, :]
    before_error = np.linalg.norm(pred_centers - gt_centers, axis=1)
    after_error = np.linalg.norm(aligned_centers - gt_centers, axis=1)
    stats["camera_center_error_before"] = {
        "mean": float(before_error.mean()),
        "median": float(np.median(before_error)),
        "max": float(before_error.max()),
    }
    stats["camera_center_error_after"] = {
        "mean": float(after_error.mean()),
        "median": float(np.median(after_error)),
        "max": float(after_error.max()),
    }
    stats["rigid_first_frame_rotation"] = rotation.tolist()
    stats["rigid_first_frame_translation"] = translation.tolist()
    return aligned_points, stats


def camera_centers_from_w2c_torch(w2c: torch.Tensor) -> torch.Tensor:
    rot = w2c[:, :3, :3]
    trans = w2c[:, :3, 3]
    return -torch.einsum("nij,nj->ni", rot.transpose(1, 2), trans)


def umeyama_similarity_torch(
    src: torch.Tensor,
    dst: torch.Tensor,
    estimate_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if src.shape != dst.shape:
        raise ValueError(f"Source and destination must have the same shape, got {tuple(src.shape)} vs {tuple(dst.shape)}")
    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected Nx3 point sets, got {tuple(src.shape)}")
    if src.shape[0] < 3:
        raise ValueError("At least 3 points are required for Sim(3) estimation.")

    src = src.to(dtype=torch.float64)
    dst = dst.to(dtype=torch.float64, device=src.device)
    src_mean = src.mean(dim=0)
    dst_mean = dst.mean(dim=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    cov = (dst_centered.T @ src_centered) / src.shape[0]
    if torch.linalg.matrix_rank(cov).item() < 2:
        raise ValueError("Point configuration is degenerate for Umeyama alignment.")

    u, singular_values, vh = torch.linalg.svd(cov)
    d = torch.eye(3, dtype=torch.float64, device=src.device)
    if (torch.linalg.det(u) * torch.linalg.det(vh)).item() < 0:
        d[-1, -1] = -1.0

    rotation = u @ d @ vh
    if estimate_scale:
        src_var = torch.mean(torch.sum(src_centered * src_centered, dim=1))
        if src_var.item() < 1e-12:
            raise ValueError("Source points have near-zero variance; scale is ill-defined.")
        scale = torch.sum(singular_values * torch.diag(d)) / src_var
    else:
        scale = torch.ones((), dtype=torch.float64, device=src.device)

    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def apply_similarity_torch(
    points: torch.Tensor,
    scale: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    points_dtype = points.dtype
    points = points.to(dtype=torch.float64)
    aligned = scale * (points @ rotation.T) + translation
    return aligned.to(dtype=points_dtype)


def rigid_transform_from_first_frame_torch(pred_w2c: torch.Tensor, gt_w2c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred_c2w = torch.linalg.inv(pred_w2c)
    gt_c2w = torch.linalg.inv(gt_w2c)
    transform = gt_c2w @ pred_w2c
    return transform[:3, :3], transform[:3, 3]


def _camera_error_stats(errors: torch.Tensor) -> dict[str, float]:
    errors = errors.detach().cpu()
    return {
        "mean": float(errors.mean()),
        "median": float(torch.median(errors)),
        "max": float(errors.max()),
    }


def align_points_to_eth3d_torch(
    world_points: torch.Tensor,
    pred_w2c: torch.Tensor,
    gt_w2c: torch.Tensor,
    alignment_mode: str,
) -> tuple[torch.Tensor, dict]:
    device = world_points.device
    pred_w2c = pred_w2c.to(device=device, dtype=torch.float64)
    gt_w2c = gt_w2c.to(device=device, dtype=torch.float64)
    pred_centers = camera_centers_from_w2c_torch(pred_w2c)
    gt_centers = camera_centers_from_w2c_torch(gt_w2c)
    stats: dict[str, object] = {
        "requested_alignment_mode": alignment_mode,
        "used_alignment_mode": alignment_mode,
        "sim3_success": False,
    }

    before_error = torch.linalg.norm(pred_centers - gt_centers, dim=1)
    stats["camera_center_error_before"] = _camera_error_stats(before_error)

    if alignment_mode == "none":
        return world_points, stats

    if alignment_mode == "sim3":
        try:
            scale, rotation, translation = umeyama_similarity_torch(pred_centers, gt_centers, estimate_scale=True)
            aligned_points = apply_similarity_torch(world_points, scale, rotation, translation)
            aligned_centers = apply_similarity_torch(pred_centers, scale, rotation, translation)
            after_error = torch.linalg.norm(aligned_centers - gt_centers, dim=1)
            stats["sim3_success"] = True
            stats["sim3_scale"] = float(scale.detach().cpu())
            stats["sim3_rotation"] = rotation.detach().cpu().tolist()
            stats["sim3_translation"] = translation.detach().cpu().tolist()
            stats["camera_center_error_after"] = _camera_error_stats(after_error)
            return aligned_points, stats
        except Exception as exc:
            logger.warning("Sim3 alignment failed: %s. Falling back to rigid_first_frame.", exc)
            stats["used_alignment_mode"] = "rigid_first_frame"
            stats["sim3_failure_reason"] = str(exc)

    rotation, translation = rigid_transform_from_first_frame_torch(pred_w2c[0], gt_w2c[0])
    aligned_points = (world_points.to(dtype=torch.float64) @ rotation.T + translation).to(dtype=world_points.dtype)
    aligned_centers = pred_centers @ rotation.T + translation
    after_error = torch.linalg.norm(aligned_centers - gt_centers, dim=1)
    stats["camera_center_error_after"] = _camera_error_stats(after_error)
    stats["rigid_first_frame_rotation"] = rotation.detach().cpu().tolist()
    stats["rigid_first_frame_translation"] = translation.detach().cpu().tolist()
    return aligned_points, stats


def build_point_cloud_torch(
    world_points: torch.Tensor,
    world_points_conf: torch.Tensor,
    images_uint8: torch.Tensor,
    point_conf_thresh: float,
    max_points_per_image: int,
    generator: torch.Generator,
) -> tuple[np.ndarray, dict]:
    num_frames = world_points.shape[0]

    if max_points_per_image <= 0:
        conf_mask = world_points_conf >= point_conf_thresh
        per_image_counts = [int(count) for count in conf_mask.reshape(num_frames, -1).sum(dim=1).detach().cpu().tolist()]
        if conf_mask.any():
            points_tensor = torch.cat([world_points[conf_mask].float(), images_uint8[conf_mask].float()], dim=1)
            points = points_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            points = np.empty((0, 6), dtype=np.float32)
        stats = {
            "per_image_kept_points": per_image_counts,
            "total_points": int(points.shape[0]),
        }
        return points, stats

    fused_points = []
    per_image_counts = []

    for frame_idx in range(num_frames):
        conf_mask = world_points_conf[frame_idx] >= point_conf_thresh
        flat_indices = torch.nonzero(conf_mask.reshape(-1), as_tuple=False).squeeze(1)
        if max_points_per_image > 0 and flat_indices.numel() > max_points_per_image:
            order = torch.randperm(flat_indices.numel(), generator=generator, device=flat_indices.device)[:max_points_per_image]
            flat_indices = torch.sort(flat_indices[order]).values
        per_image_counts.append(int(len(flat_indices)))
        if flat_indices.numel() == 0:
            continue

        xyz = world_points[frame_idx].reshape(-1, 3)[flat_indices]
        rgb = images_uint8[frame_idx].reshape(-1, 3)[flat_indices]
        fused_points.append(torch.cat([xyz.float(), rgb.float()], dim=1))

    if fused_points:
        points = torch.cat(fused_points, dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        points = np.empty((0, 6), dtype=np.float32)

    stats = {
        "per_image_kept_points": per_image_counts,
        "total_points": int(points.shape[0]),
    }
    return points, stats


def scene_results_dir(results_path: Path, scene_name: str) -> Path:
    output_dir = results_path / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_prediction(image_paths: list[Path], model: VGGT) -> dict:
    predictions = predict(image_paths, model)
    return {
        "world_points": predictions["world_points"].detach(),
        "world_points_conf": predictions["world_points_conf"].detach(),
        "pose_enc": predictions["pose_enc"].detach(),
        "images": predictions["images"].detach(),
        "image_names": [path.name for path in image_paths],
        "image_relative_names": [str(path) for path in image_paths],
        "images_shape_hw": tuple(predictions["images"].shape[-2:]),
    }


def export_scene(
    scene_dir: Path,
    scene_records: list[dict],
    predictions_payload: dict,
    scene_info: dict,
    args: argparse.Namespace,
) -> dict:
    scene_name = scene_dir.name
    output_dir = scene_results_dir(args.results_path, scene_name)

    world_points = predictions_payload["world_points"]
    world_points_conf = predictions_payload["world_points_conf"]
    if not isinstance(world_points, torch.Tensor) or not isinstance(world_points_conf, torch.Tensor):
        raise TypeError("Prediction payload must keep world_points and world_points_conf as torch tensors.")
    world_points = world_points[0]
    world_points_conf = world_points_conf[0]

    image_paths = [record["image_path"] for record in scene_records]
    pred_relative_names = predictions_payload["image_relative_names"]
    expected_relative_names = [str(path) for path in image_paths]
    if list(pred_relative_names) != expected_relative_names:
        raise ValueError(
            f"Prediction image order for scene {scene_name} does not match current scene records.\n"
            f"Expected: {expected_relative_names[:3]}...\n"
            f"Found: {list(pred_relative_names)[:3]}..."
        )

    pred_w2c = decode_prediction_extrinsics(predictions_payload)
    gt_w2c = torch.as_tensor(
        np.stack([record["w2c"] for record in scene_records], axis=0),
        dtype=torch.float64,
        device=world_points.device,
    )

    aligned_world_points, alignment_stats = align_points_to_eth3d_torch(
        world_points=world_points,
        pred_w2c=pred_w2c,
        gt_w2c=gt_w2c,
        alignment_mode=args.alignment_mode,
    )

    images_uint8 = prediction_images_uint8(predictions_payload["images"], device=world_points.device)
    expected_hw = tuple(world_points_conf.shape[-2:])
    if images_uint8.shape[1:3] != expected_hw:
        raise ValueError(
            f"Preprocessed RGB size for scene {scene_name} does not match prediction grid: "
            f"RGB HxW={images_uint8.shape[1:3]}, prediction HxW={expected_hw}. "
            "Check the model input preprocessing path."
        )

    generator = torch.Generator(device=world_points.device)
    generator.manual_seed(args.seed + sum(ord(ch) for ch in scene_name))
    points, point_stats = build_point_cloud_torch(
        aligned_world_points,
        world_points_conf,
        images_uint8,
        point_conf_thresh=args.point_conf_thresh,
        max_points_per_image=args.max_points_per_image,
        generator=generator,
    )

    ply_path = args.results_path / f"{scene_name}.ply"
    write_ply(ply_path, points)

    metadata = {
        "scene_name": scene_name,
        "usable_images": len(scene_records),
        "missing_images": int(scene_info["missing_images"]),
        "point_conf_thresh": float(args.point_conf_thresh),
        "max_points_per_image": int(args.max_points_per_image),
        "output_ply": str(ply_path),
        "alignment": alignment_stats,
        "point_stats": point_stats,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def prediction_images_uint8(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    if images.ndim == 5:
        images = images[0]
    if images.ndim != 4:
        raise ValueError(f"Expected prediction images with shape Nx3xHxW or 1xNx3xHxW, got {tuple(images.shape)}")
    images = images.to(device=device)
    images = images.permute(0, 2, 3, 1).mul(255.0).round().clamp(0, 255)
    return images.to(torch.uint8)


def process_scene(scene_dir: Path, model: VGGT, args: argparse.Namespace) -> Optional[dict]:
    scene_name = scene_dir.name
    logger.info("Processing ETH3D scene %s", scene_name)

    scene_records, scene_info = collect_scene_records(scene_dir)
    if len(scene_records) < 2:
        logger.warning(
            "Scene %s has only %d usable images after filtering missing files, skipping.",
            scene_name,
            len(scene_records),
        )
        return {
            "scene_name": scene_name,
            "status": "skipped",
            "reason": "fewer_than_two_usable_images",
            **scene_info,
        }

    image_paths = [record["image_path"] for record in scene_records]

    predictions_payload = run_prediction(image_paths, model)
    predictions_payload["total_images_in_txt"] = scene_info["total_images_in_txt"]

    metadata = export_scene(scene_dir, scene_records, predictions_payload, scene_info, args)
    metadata.update(scene_info)
    metadata["status"] = "ok"
    logger.info("Finished scene %s, wrote %s with %d points", scene_name, metadata["output_ply"], metadata["point_stats"]["total_points"])
    return metadata


def run_alignment_selfcheck() -> None:
    src = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    scale_gt = 2.5
    rotation_gt = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    translation_gt = np.array([3.0, -2.0, 5.0], dtype=np.float64)
    dst = apply_similarity(src, scale_gt, rotation_gt, translation_gt)
    scale_pred, rotation_pred, translation_pred = umeyama_similarity(src, dst, estimate_scale=True)

    if not np.allclose(scale_pred, scale_gt, atol=1e-5):
        raise AssertionError(f"Sim3 self-check failed for scale: {scale_pred} vs {scale_gt}")
    if not np.allclose(rotation_pred, rotation_gt, atol=1e-5):
        raise AssertionError("Sim3 self-check failed for rotation.")
    if not np.allclose(translation_pred, translation_gt, atol=1e-5):
        raise AssertionError("Sim3 self-check failed for translation.")

    pred_w2c = np.eye(4, dtype=np.float64)
    gt_c2w = np.eye(4, dtype=np.float64)
    gt_c2w[:3, :3] = rotation_gt
    gt_c2w[:3, 3] = translation_gt
    gt_w2c = np.linalg.inv(gt_c2w)
    rigid_rotation, rigid_translation = rigid_transform_from_first_frame(pred_w2c, gt_w2c)
    transformed = src @ rigid_rotation.T + rigid_translation[None, :]
    expected = src @ rotation_gt.T + translation_gt[None, :]
    if not np.allclose(transformed, expected, atol=1e-5):
        raise AssertionError("Rigid first-frame self-check failed.")


def main() -> None:
    configure_logging()
    logger.setLevel(logging.INFO)
    args = parse_args()

    if args.max_points_per_image < 0:
        raise ValueError("--max_points_per_image must be >= 0.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_alignment_selfcheck()

    scene_dirs = [scene_dir for scene_dir in discover_scene_dirs(args.eth3d_root, args.scenes) if validate_scene_dir(scene_dir)]
    if not scene_dirs:
        raise ValueError("No valid ETH3D scenes found to process.")

    args.results_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from %s", args.model_path)
    model = load_model(
        args.model_path,
        model_args={
            "enable_camera": True,
            "enable_point": True,
            "enable_depth": False,
            "enable_track": False,
        },
        backend=args.backend,
        quant_config={
            "smoothquant_path": args.smoothquant_scale_path,
            "smoothquant_strict": not args.smoothquant_allow_missing,
        },
    )

    summary = []
    try:
        for scene_dir in scene_dirs:
            try:
                result = process_scene(scene_dir, model, args)
                if result is not None:
                    summary.append(result)
            except Exception as exc:
                logger.exception("Failed to process scene %s", scene_dir.name)
                summary.append(
                    {
                        "scene_name": scene_dir.name,
                        "status": "failed",
                        "reason": str(exc),
                    }
                )
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = args.results_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
