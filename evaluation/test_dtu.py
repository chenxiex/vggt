import torch
from vggt.models.vggt import VGGT
import random
from pathlib import Path
import os
import numpy as np
from PIL import Image
import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional
from torch.multiprocessing.spawn import spawn
import torch.nn.functional as F

from utils import load_model, predict, read_pfm, upsample_image, upsample_images, write_ply, open3d_filter

logger = logging.getLogger(__name__)

MAX_IO_WORKERS = max(1, min(8, os.cpu_count() or 1))
MAX_UPSAMPLE_WORKERS = max(1, min(8, os.cpu_count() or 1))


def configure_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_gt_depth(gt_depths_path: Path, sample_no: list[int]):
    '''
    Args:
        gt_depths_path: 真实深度图所在的文件夹路径。gt_depths_path/f"depth_map_{i:04}.pfm"
        sample_no: 需要加载的真实深度图对应的编号列表
    Returns:
        gt_depth: 形状为 (batch_size, H, W)，
    '''
    sampled_gt_depth_paths = [gt_depths_path /
                              f"depth_map_{i:04}.pfm" for i in sample_no]

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
    """
    Aligns a predicted depth map to a ground truth depth map using scale and shift.
    The alignment is: gt_aligned_to_pred ≈ scale * pred_depth + shift.

    Args:
        pred_depth (np.ndarray): The HxW predicted depth map.
        gt_depth (np.ndarray): The HxW ground truth depth map.
        valid_mask: (np.ndarray): A boolean mask of the valid pixels in the depth maps.
        min_valid_pixels (int): The minimum number of valid pixels required for alignment.

    Returns:
        tuple[float, float, np.ndarray]:
            - scale (float): The calculated scale factor. (NaN if alignment failed)
            - shift (float): The calculated shift offset. (NaN if alignment failed)
    """
    if pred_depth.shape != gt_depth.shape:
        raise ValueError(
            f"Predicted depth shape {pred_depth.shape} must match GT depth shape {gt_depth.shape}"
        )

    # Extract valid depth values
    gt_masked = gt_depth[valid_mask]
    pred_masked = pred_depth[valid_mask]

    if len(gt_masked) < min_valid_pixels:
        logger.warning(
            f"Warning: Not enough valid pixels ({len(gt_masked)} < {min_valid_pixels}) to align. "
            "Using all pixels."
        )
        gt_masked = gt_depth.reshape(-1)
        pred_masked = pred_depth.reshape(-1)

    # Handle case where pred_masked has no variance (e.g., all zeros or a constant value)
    if np.std(pred_masked) < 1e-6:  # Small epsilon to check for near-constant values
        logger.warning(
            "Warning: Predicted depth values in the valid mask have near-zero variance. "
            "Scale is ill-defined. Setting scale=1 and solving for shift only."
        )
        scale = 1.0
        # or np.median(gt_masked) - np.median(pred_masked)
        shift = np.mean(gt_masked) - np.mean(pred_masked)
    else:
        A = np.vstack([pred_masked, np.ones_like(pred_masked)]).T
        try:
            x, residuals, rank, s_values = np.linalg.lstsq(
                A, gt_masked, rcond=None)
            scale, shift = x[0], x[1]
        except np.linalg.LinAlgError as e:
            logger.warning(
                f"Warning: Least squares alignment failed ({e}). Returning original prediction.")
            return np.nan, np.nan

    return scale, shift


def parse_cam(cam_file: Path):
    with open(cam_file, "r", encoding="utf-8") as fp:
        cam_txt = fp.readlines()

    def _parse_rows(xs):
        return list(map(lambda x: list(map(float, x.strip().split())), xs))

    extr_mat = _parse_rows(cam_txt[1:5])
    intr_mat = _parse_rows(cam_txt[7:10])

    extr_mat = np.array(extr_mat, np.float32)
    intr_mat = np.array(intr_mat, np.float32)

    return extr_mat, intr_mat


def load_data(dtu_test_1200_path: Path, scene_name: str, sample_no: list[int]):
    def _load_single_view(view: int):
        img_file = dtu_test_1200_path / f"Rectified/{scene_name}/rect_{view + 1:03d}_3_r5000.png"
        cam_file = dtu_test_1200_path / f"Cameras/{view:08}_cam.txt"

        extr_mat, intr_mat = parse_cam(cam_file)
        proj_mat = np.eye(4)
        proj_mat[:3, :4] = intr_mat[:3, :3] @ extr_mat[:3, :4]
        rgb = np.array(Image.open(img_file))
        return torch.from_numpy(proj_mat), rgb

    if len(sample_no) <= 1:
        loaded = [_load_single_view(view) for view in sample_no]
    else:
        with ThreadPoolExecutor(max_workers=min(MAX_IO_WORKERS, len(sample_no))) as executor:
            loaded = list(executor.map(_load_single_view, sample_no))

    proj_list, rgbs = zip(*loaded)
    projs = torch.stack(list(proj_list)).float()

    # 归一化，维度从[H, W, C]调整为[C, H, W]
    rgb_tensors = [torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1)
                   for img in rgbs]
    rgbs = torch.stack(rgb_tensors)           # (B,3,H,W)

    return projs, rgbs


def upsample_images_parallel(images: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
    num_images = int(images.shape[0])
    max_workers = min(MAX_UPSAMPLE_WORKERS, num_images)
    if max_workers <= 1:
        return upsample_images(images, target_w, target_h)

    upsample_one = partial(upsample_image, target_w=target_w, target_h=target_h)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        upsampled_images = list(executor.map(upsample_one, images))

    return torch.stack(upsampled_images, dim=0)


def scale_proj_matrices(projs: torch.Tensor, scale_x: float, scale_y: float) -> torch.Tensor:
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError(f"scale_x and scale_y must be positive, got {scale_x} and {scale_y}")

    scaled_projs = projs.clone()
    scaled_projs[:, 0, :] *= scale_x
    scaled_projs[:, 1, :] *= scale_y
    return scaled_projs


def resize_rgb_images(images: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    return F.interpolate(images, size=(target_h, target_w), mode="bilinear", align_corners=False)


def normalize_depth_tensor(depths: torch.Tensor) -> torch.Tensor:
    if depths.ndim == 4 and depths.shape[-1] == 1:
        return depths.squeeze(-1)
    if depths.ndim != 3:
        raise ValueError(f"Expected depth tensor with shape (S,H,W) or (S,H,W,1), got {tuple(depths.shape)}")
    return depths


def normalize_conf_tensor(conf: torch.Tensor) -> torch.Tensor:
    if conf.ndim == 4 and conf.shape[-1] == 1:
        return conf.squeeze(-1)
    if conf.ndim != 3:
        raise ValueError(f"Expected confidence tensor with shape (S,H,W) or (S,H,W,1), got {tuple(conf.shape)}")
    return conf


def build_scene_names(dtu_test_1200_path: Path, scans: Optional[str]) -> list[str]:
    if not scans or scans.lower() == "true":
        with open(dtu_test_1200_path/"scan_list_test.txt") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    scene_ids = [scan_id.strip() for scan_id in scans.split(',') if scan_id.strip()]
    return [f"scan{scene_id}" for scene_id in scene_ids]


def build_sample_indices(scene_name: str, sample_size: int, base_seed: int) -> list[int]:
    scene_digits = "".join(ch for ch in scene_name if ch.isdigit())
    if scene_digits:
        seed_offset = int(scene_digits)
    else:
        seed_offset = sum(ord(ch) for ch in scene_name)

    rng = random.Random(base_seed + seed_offset)
    return rng.sample(range(0, 49), sample_size)


def parse_gpu_ids(gpu_ids_arg: Optional[str]) -> list[int]:
    if not torch.cuda.is_available():
        if gpu_ids_arg:
            logger.warning(
                "CUDA is not available, ignoring --gpu_ids=%s and running on CPU.",
                gpu_ids_arg,
            )
        return []

    device_count = torch.cuda.device_count()
    if gpu_ids_arg is None or gpu_ids_arg.lower() == "auto":
        return list(range(device_count))

    gpu_ids = []
    for token in gpu_ids_arg.split(','):
        token = token.strip()
        if not token:
            continue
        gpu_id = int(token)
        if gpu_id < 0 or gpu_id >= device_count:
            raise ValueError(
                f"GPU id {gpu_id} is out of range. Available ids: 0..{device_count - 1}."
            )
        if gpu_id not in gpu_ids:
            gpu_ids.append(gpu_id)

    if not gpu_ids:
        raise ValueError("No valid GPU ids were provided in --gpu_ids.")

    return gpu_ids


def split_scene_names(scene_names: list[str], num_workers: int) -> list[list[str]]:
    return [scene_names[i::num_workers] for i in range(num_workers)]


def process_scene(
    args: argparse.Namespace,
    scene_name: str,
    model: VGGT,
    worker_tag: str,
):
    logger.info("%s Processing %s...", worker_tag, scene_name)
    logger.info("%s Predicting depth maps...", worker_tag)
    images_path = args.dtu_test_1200_path/"Rectified"/scene_name
    sample_no = build_sample_indices(scene_name, args.sample_size, args.seed)
    sampled_image_paths = [
        images_path/f"rect_{i+1:03d}_3_r5000.png" for i in sample_no]
    predictions = predict(sampled_image_paths, model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 对齐
    logger.info("%s Aligning predicted depth maps to ground truth...", worker_tag)
    gt_depths_path = args.dtu_depths_path/"Depths"/scene_name
    gt_depth = load_gt_depth(gt_depths_path, sample_no)
    gt_depth_w, gt_depth_h = gt_depth[0].shape[:2]
    lowres_depths = normalize_depth_tensor(predictions['depth'][0])
    lowres_conf = normalize_conf_tensor(predictions['depth_conf'][0])
    upsampled_pred_depth = upsample_images_parallel(
        lowres_depths, gt_depth_w, gt_depth_h)
    upsampled_depth_conf = upsample_images_parallel(
        lowres_conf, gt_depth_w, gt_depth_h)

    valid_mask = (gt_depth > 1e-3) & (upsampled_depth_conf > 3)

    align_mask = valid_mask.reshape(-1)
    align_depth_map = upsampled_pred_depth.reshape(-1)
    align_gt_depth = gt_depth.reshape(-1)

    scale_val, shift_val = align_pred_to_gt(
        align_depth_map.cpu().numpy(),
        align_gt_depth.cpu().numpy(),
        align_mask.cpu().numpy()
    )

    scale = torch.tensor(scale_val, dtype=torch.float32)
    shift = torch.tensor(shift_val, dtype=torch.float32)

    aligned_upsampled_depth = upsampled_pred_depth * scale + shift

    # 点云融合
    logger.info("%s Fusing depth maps into point cloud and saving results...", worker_tag)
    projs, rgbs = load_data(args.dtu_test_1200_path, scene_name, sample_no)
    if args.upsample_align_only:
        depth_h, depth_w = lowres_depths.shape[-2:]
        rgb_h, rgb_w = rgbs.shape[-2:]
        scale_y = depth_h / rgb_h
        scale_x = depth_w / rgb_w

        depths = (lowres_depths * scale + shift) * (lowres_conf > 3)
        depths = depths.unsqueeze(1)
        projs = scale_proj_matrices(projs, scale_x=scale_x, scale_y=scale_y)
        rgbs = resize_rgb_images(rgbs, depth_h, depth_w)
    else:
        depths = aligned_upsampled_depth * (upsampled_depth_conf > 3)
        depths = depths.unsqueeze(1)

    if torch.cuda.is_available():
        fusion_device = torch.device("cuda", torch.cuda.current_device())
        depths = depths.to(fusion_device, non_blocking=True)
        projs = projs.to(fusion_device, non_blocking=True)
        rgbs = rgbs.to(fusion_device, non_blocking=True)
    points, point_scores = open3d_filter(
        depths,
        projs,
        rgbs,
        dist_thresh=args.dist_thresh,
        batch_size=args.fusion_batch_size,
        num_consist=args.num_consist,
        return_point_scores=True,
    )
    write_ply(args.results_path /
            f"{int(scene_name[4:]):03d}.ply", points)
    logger.info(
        "%s Finished processing %s, written to %03d.ply with %d fused points",
        worker_tag,
        scene_name,
        int(scene_name[4:]),
        int(points.shape[0]),
    )


def run_worker(
    args: argparse.Namespace,
    scene_names: list[str],
    worker_idx: int,
    gpu_id: Optional[int],
):
    configure_logging()
    logger.setLevel(logging.INFO)

    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        worker_tag = f"[worker {worker_idx} | cuda:{gpu_id}]"
        logger.info("%s Bound to GPU %s", worker_tag, torch.cuda.get_device_name(gpu_id))
    else:
        worker_tag = f"[worker {worker_idx} | cpu]"
        logger.info("%s Running on CPU", worker_tag)

    if not scene_names:
        logger.info("%s No scenes assigned, exiting.", worker_tag)
        return

    logger.info("%s Using model from %s", worker_tag, args.model_path)
    model = load_model(args.model_path, model_args={"enable_point": False, "enable_track": False})

    try:
        for scene_name in scene_names:
            process_scene(args, scene_name, model, worker_tag)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _worker_entry(
    worker_idx: int,
    gpu_ids: list[int],
    scene_chunks: list[list[str]],
    args: argparse.Namespace,
):
    gpu_id = gpu_ids[worker_idx] if gpu_ids else None
    run_worker(args, scene_chunks[worker_idx], worker_idx, gpu_id)


if __name__ == "__main__":
    configure_logging()
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtu_test_1200_path", type=Path,
                        required=True, help="Path to the DTU testing dataset")
    parser.add_argument("--dtu_depths_path", type=Path,
                        required=True, help="Path to the DTU raw depth maps")
    parser.add_argument("--results_path", type=Path, required=True,
                        help="Path to save the DTU testing results")
    parser.add_argument("--model_path", type=Path,
                        required=True, help="Path to the trained VGGT model")
    parser.add_argument("--sample_size", type=int, default=49,
                        help="Sample size for prediction")
    parser.add_argument('--scans', type=str, default=None,
                        help="Scene ID numbers to evaluate (e.g., 1,2,3). If not provided or set to 'true', will evaluate all scenes in the scan list.")
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help="Comma-separated GPU ids for scene-level parallelism (e.g., 0,1,2). Default is 'auto' (all visible GPUs).")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for deterministic per-scene image sampling.")
    parser.add_argument('--dist_thresh', type=float, default=1.0,
                        help="Distance threshold for point cloud fusion consistency check.")
    parser.add_argument('--num_consist', type=int, default=4,
                        help="Minimum number of consistent depth maps required to keep a point in fusion.")
    parser.add_argument('--fusion_batch_size', type=int, default=20,
                        help="Batch size for point cloud fusion to balance memory usage and speed.")
    parser.add_argument(
        '--upsample_align_only',
        type=bool,
        default=True,
        help="Use upsampled depth only for GT alignment; keep fusion on the native prediction resolution and scale camera intrinsics accordingly. Enabled by default.",
    )
    args = parser.parse_args()

    if args.sample_size <= 0 or args.sample_size > 49:
        raise ValueError("sample_size must be in [1, 49].")

    random.seed(args.seed)

    scene_names = build_scene_names(args.dtu_test_1200_path, args.scans)
    if not scene_names:
        raise ValueError("No scenes found to evaluate.")

    gpu_ids = parse_gpu_ids(args.gpu_ids)

    if len(gpu_ids) > 1 and len(scene_names) > 1:
        logger.info("Running scene-level multi-GPU parallel evaluation on GPUs: %s", gpu_ids)
        scene_chunks = split_scene_names(scene_names, len(gpu_ids))
        spawn(
            _worker_entry,
            args=(gpu_ids, scene_chunks, args),
            nprocs=len(gpu_ids),
            join=True,
        )
    else:
        selected_gpu = gpu_ids[0] if gpu_ids else None
        if selected_gpu is not None:
            logger.info("Running single-worker evaluation on GPU %d", selected_gpu)
        else:
            logger.info("Running single-worker evaluation on CPU")
        run_worker(args, scene_names, worker_idx=0, gpu_id=selected_gpu)
