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
from typing import Optional
from torch.multiprocessing.spawn import spawn

from utils import load_model, predict, read_pfm, upsample_images, write_ply, open3d_filter

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


def save_predictions(results_path: Path, scene_name: str, predictions, sample_no):
    save_dict = {
        "predictions": predictions,
        "sample_no": sample_no
    }

    if not results_path.exists():
        os.makedirs(results_path, exist_ok=True)

    torch.save(save_dict, results_path/f"{scene_name}.pt")


def load_predictions(results_path: Path, scene_name: str):
    results = torch.load(
        results_path/f"{scene_name}.pt", map_location=torch.device("cpu"))
    sample_no = results["sample_no"]
    predictions = results["predictions"]
    return predictions, sample_no


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

    gt_depth = []

    for gt_depth_path in sampled_gt_depth_paths:
        data, scale = read_pfm(gt_depth_path)
        gt_depth.append(data*scale)

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
    cam_txt = open(cam_file).readlines()
    def f(xs): return list(map(lambda x: list(map(float, x.strip().split())), xs))

    extr_mat = f(cam_txt[1:5])
    intr_mat = f(cam_txt[7:10])

    extr_mat = np.array(extr_mat, np.float32)
    intr_mat = np.array(intr_mat, np.float32)

    return extr_mat, intr_mat


def load_data(dtu_test_1200_path: Path, scene_name: str, sample_no: list[int]):

    projs = []
    rgbs = []

    for view in sample_no:
        img_file = dtu_test_1200_path / \
            f"Rectified/{scene_name}/rect_{view+1:03d}_3_r5000.png"
        cam_file = dtu_test_1200_path/f"Cameras/{view:08}_cam.txt"

        extr_mat, intr_mat = parse_cam(cam_file)
        proj_mat = np.eye(4)
        proj_mat[:3, :4] = intr_mat[:3, :3] @ extr_mat[:3, :4]
        projs.append(torch.from_numpy(proj_mat))

        rgb = np.array(Image.open(img_file))
        rgbs.append(rgb)

    projs = torch.stack(projs).float()

    # 归一化，维度从[H, W, C]调整为[C, H, W]
    rgb_tensors = [torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1)
                   for img in rgbs]
    rgbs = torch.stack(rgb_tensors)           # (B,3,H,W)

    return projs, rgbs


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
    model: Optional[VGGT],
    worker_tag: str,
):
    logger.info("%s Processing %s...", worker_tag, scene_name)
    if not args.no_pred:
        logger.info("%s Predicting depth maps...", worker_tag)
        images_path = args.dtu_test_1200_path/"Rectified"/scene_name
        sample_no = build_sample_indices(scene_name, args.sample_size, args.seed)
        sampled_image_paths = [
            images_path/f"rect_{i+1:03d}_3_r5000.png" for i in sample_no]
        assert model is not None, "Model should not be None when not skipping prediction"
        predictions = predict(sampled_image_paths, model)
        save_predictions(args.results_path, scene_name,
                         predictions, sample_no)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        logger.info("%s Loading predictions...", worker_tag)
        predictions, sample_no = load_predictions(
            args.results_path, scene_name)

    if args.pred_only:
        return

    # 对齐
    logger.info("%s Aligning predicted depth maps to ground truth...", worker_tag)
    gt_depths_path = args.dtu_depths_path/"Depths"/scene_name
    gt_depth = load_gt_depth(gt_depths_path, sample_no)
    gt_depth_w, gt_depth_h = gt_depth[0].shape[:2]
    depths = predictions['depth'][0]
    conf = predictions['depth_conf'][0]
    upsampled_pred_depth = upsample_images(
        depths, gt_depth_w, gt_depth_h)
    upsampled_depth_conf = upsample_images(
        conf, gt_depth_w, gt_depth_h)

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
    depths = aligned_upsampled_depth * (upsampled_depth_conf > 3)
    depths = depths.unsqueeze(1)

    # 点云融合
    logger.info("%s Fusing depth maps into point cloud and saving results...", worker_tag)
    projs, rgbs = load_data(args.dtu_test_1200_path, scene_name, sample_no)
    points = open3d_filter(depths, projs, rgbs,
                        dist_thresh=1.0, batch_size=20, num_consist=4)
    write_ply(args.results_path /
            f"{int(scene_name[4:]):03d}.ply", points)
    logger.info("%s Finished processing %s, written to %03d.ply",
                worker_tag, scene_name, int(scene_name[4:]))


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

    model = None
    if not args.no_pred:
        logger.info("%s Using model from %s", worker_tag, args.model_path)
        model = load_model(args.model_path, model_args={"enable_point": False, "enable_track": False})

    try:
        for scene_name in scene_names:
            process_scene(args, scene_name, model, worker_tag)
    finally:
        if model is not None:
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
                        required=False, help="Path to the trained VGGT model")
    parser.add_argument("--sample_size", type=int, default=49,
                        help="Sample size for prediction")
    parser.add_argument("--no_pred", action="store_true",
                        help="If set, skip prediction and only load existing predictions")
    parser.add_argument("--pred_only", action="store_true",
                        help="If set, only perform prediction without alignment and fusion")
    parser.add_argument('--scans', type=str, default=None,
                        help="Scene ID numbers to evaluate (e.g., 1,2,3). If not provided or set to 'true', will evaluate all scenes in the scan list.")
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help="Comma-separated GPU ids for scene-level parallelism (e.g., 0,1,2). Default is 'auto' (all visible GPUs).")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for deterministic per-scene image sampling.")
    args = parser.parse_args()

    if not args.no_pred and not args.model_path:
        raise ValueError(
            "Model path must be provided if not skipping prediction.")

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
