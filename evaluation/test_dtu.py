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
            f"Rectified/{scene_name}/rect_{view:03d}_3_r5000.png"
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
    parser.add_argument("--scans", type=int, nargs='+', required=False,
                        help="Scene ID numbers to evaluate (e.g., 1 2 3)")
    args = parser.parse_args()

    if not args.no_pred and not args.model_path:
        raise ValueError(
            "Model path must be provided if not skipping prediction.")

    if args.scans:
        scene_names = [f"scan{i}" for i in args.scans]
    else:
        with open(args.dtu_test_1200_path/"scan_list_test.txt") as f:
            scene_names = [line.strip() for line in f.readlines()]

    for scene_name in scene_names:
        # 推理
        logger.info(f"Processing {scene_name}...")
        if not args.no_pred:
            logger.info("Predicting depth maps...")
            model = load_model(args.model_path)
            images_path = args.dtu_test_1200_path/"Rectified"/scene_name
            sample_no = random.sample(range(1, 50), args.sample_size)
            sampled_image_paths = [
                images_path/f"rect_{i:03d}_3_r5000.png" for i in sample_no]
            predictions = predict(sampled_image_paths, model)
            save_predictions(args.results_path, scene_name,
                             predictions, sample_no)
            del model
            torch.cuda.empty_cache()
        else:
            logger.info("Loading predictions...")
            predictions, sample_no = load_predictions(
                args.results_path, scene_name)

        # 对齐
        logger.info("Aligning predicted depth maps to ground truth...")
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
        logger.info("Fusing depth maps into point cloud and saving results...")
        projs, rgbs = load_data(args.dtu_test_1200_path, scene_name, sample_no)
        points = open3d_filter(depths, projs, rgbs,
                               dist_thresh=1.0, batch_size=20, num_consist=4)
        write_ply(args.results_path /
                  f"vggt{int(scene_name[4:]):03d}_l3.ply", points)
        logger.info(f"Finished processing {scene_name}, written to " +
                    f"vggt{int(scene_name[4:]):03d}_l3.ply")
