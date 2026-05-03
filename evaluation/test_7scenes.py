import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from utils import load_model, predict
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VGGT camera pose on 7-scenes")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data/7-scenes"),
        help="Root path of 7-scenes dataset",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("ckpt/model.pt"),
        help="Path to VGGT model checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "train"],
        default="test",
        help="Which split file to use (TestSplit.txt or TrainSplit.txt)",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="all",
        help="Comma-separated scene names (e.g. chess,fire). Use 'all' for all available scenes.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=10,
        help="Number of frames sampled per sequence. <=0 means using all frames.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic frame sampling",
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
        "--output_json",
        type=Path,
        default=None,
        help="Optional path to save evaluation results in JSON format",
    )
    parser.add_argument(
        "--save_pair_errors",
        action="store_true",
        help="If set, include per-pair rotation/translation errors in output JSON.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="pseudo",
        choices=["pseudo", "torchao", "bitsandbytes", "trt_int8"],
        help="Quantization backend to use.",
    )
    return parser.parse_args()


def parse_sequence_name(raw_name: str) -> Optional[str]:
    token = raw_name.strip()
    if not token:
        return None

    digits = "".join(ch for ch in token if ch.isdigit())
    if not digits:
        logger.warning("Cannot parse sequence id from line: '%s'", token)
        return None
    return f"seq-{int(digits):02d}"


def discover_scenes(data_root: Path, scenes_arg: str) -> list[str]:
    if not data_root.exists():
        raise FileNotFoundError(f"7-scenes root path does not exist: {data_root}")

    all_scenes = sorted([p.name for p in data_root.iterdir() if p.is_dir()])
    if scenes_arg.lower() == "all":
        return all_scenes

    requested = [token.strip() for token in scenes_arg.split(",") if token.strip()]
    valid = [scene for scene in requested if (data_root / scene).is_dir()]
    missing = sorted(set(requested) - set(valid))
    for scene in missing:
        logger.warning("Scene '%s' does not exist under %s, skipping.", scene, data_root)
    return sorted(valid)


def discover_sequences(scene_dir: Path, split: str) -> list[Path]:
    split_file = scene_dir / ("TestSplit.txt" if split == "test" else "TrainSplit.txt")
    if not split_file.exists():
        logger.warning("Missing split file: %s", split_file)
        return []

    sequence_dirs: list[Path] = []
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            seq_name = parse_sequence_name(line)
            if seq_name is None:
                continue
            seq_dir = scene_dir / seq_name
            if seq_dir.is_dir():
                sequence_dirs.append(seq_dir)
            else:
                logger.warning("Sequence directory listed in split but missing: %s", seq_dir)

    return sorted(sequence_dirs)


def frame_id_from_color_path(color_path: Path) -> int:
    match = re.match(r"frame-(\d+)\.color\.png", color_path.name)
    if match is None:
        raise ValueError(f"Unexpected frame filename format: {color_path.name}")
    return int(match.group(1))


def collect_frame_records(sequence_dir: Path) -> list[dict]:
    records: list[dict] = []
    for color_path in sorted(sequence_dir.glob("frame-*.color.png")):
        pose_path = Path(str(color_path).replace(".color.png", ".pose.txt"))
        if not pose_path.exists():
            logger.warning("Missing pose file for frame %s", color_path)
            continue
        records.append(
            {
                "frame_id": frame_id_from_color_path(color_path),
                "color_path": color_path,
                "pose_path": pose_path,
            }
        )

    return records


def sample_records(records: list[dict], scene_name: str, sequence_name: str, num_frames: int, seed: int) -> list[dict]:
    if num_frames <= 0 or len(records) <= num_frames:
        return records

    scene_token = f"{scene_name}/{sequence_name}"
    scene_seed = seed + sum(ord(ch) for ch in scene_token)
    rng = random.Random(scene_seed)
    indices = sorted(rng.sample(range(len(records)), num_frames))
    return [records[idx] for idx in indices]


def load_gt_w2c(pose_paths: list[Path]) -> torch.Tensor:
    c2w_mats = []
    for pose_path in pose_paths:
        pose = np.loadtxt(pose_path, dtype=np.float32)
        if pose.shape != (4, 4):
            raise ValueError(f"Invalid pose matrix shape {pose.shape} in {pose_path}")
        if not np.isfinite(pose).all():
            raise ValueError(f"Pose matrix has non-finite values in {pose_path}")
        c2w_mats.append(pose)

    # 7-scenes pose files are camera-to-world; convert to world-to-camera.
    c2w = np.stack(c2w_mats, axis=0)
    w2c = np.linalg.inv(c2w)
    return torch.from_numpy(w2c).float()


def decode_pred_w2c(predictions: dict) -> torch.Tensor:
    if "pose_enc" not in predictions:
        raise KeyError("Model predictions do not contain 'pose_enc'.")
    if "images" not in predictions:
        raise KeyError("Model predictions do not contain 'images' needed for pose decoding.")

    image_hw = tuple(predictions["images"].shape[-2:])
    pred_extrinsic, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], image_hw)
    pred_extrinsic = pred_extrinsic[0].detach().cpu().float()  # Sx3x4

    num_frames = pred_extrinsic.shape[0]
    pred_w2c = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(num_frames, 1, 1)
    pred_w2c[:, :3, :4] = pred_extrinsic
    return pred_w2c


def build_pair_index(num_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = torch.combinations(torch.arange(num_frames), r=2)
    return pairs[:, 0], pairs[:, 1]


def rotation_error_deg(rot_gt: torch.Tensor, rot_pred: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    rel_rot = torch.matmul(rot_gt.transpose(-1, -2), rot_pred)
    trace = rel_rot[:, 0, 0] + rel_rot[:, 1, 1] + rel_rot[:, 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(min=-1.0 + eps, max=1.0 - eps)
    return torch.rad2deg(torch.arccos(cos_theta))


def translation_direction_error_deg(
    trans_gt: torch.Tensor,
    trans_pred: torch.Tensor,
    ambiguity: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    gt_norm = torch.linalg.norm(trans_gt, dim=-1, keepdim=True)
    pred_norm = torch.linalg.norm(trans_pred, dim=-1, keepdim=True)
    valid = (gt_norm > eps) & (pred_norm > eps)

    gt_unit = torch.where(valid, trans_gt / (gt_norm + eps), torch.zeros_like(trans_gt))
    pred_unit = torch.where(valid, trans_pred / (pred_norm + eps), torch.zeros_like(trans_pred))

    cos_theta = torch.sum(gt_unit * pred_unit, dim=-1).clamp(min=-1.0, max=1.0)
    ang = torch.rad2deg(torch.arccos(cos_theta))
    ang = torch.where(valid.squeeze(-1), ang, torch.full_like(ang, 180.0))

    if ambiguity:
        ang = torch.minimum(ang, torch.abs(180.0 - ang))

    return ang


def se3_to_relative_pose_error(pred_w2c: torch.Tensor, gt_w2c: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    num_frames = pred_w2c.shape[0]
    if num_frames < 2:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    idx_i, idx_j = build_pair_index(num_frames)

    rel_gt = torch.matmul(gt_w2c[idx_i], torch.linalg.inv(gt_w2c[idx_j]))
    rel_pred = torch.matmul(pred_w2c[idx_i], torch.linalg.inv(pred_w2c[idx_j]))

    r_err = rotation_error_deg(rel_gt[:, :3, :3], rel_pred[:, :3, :3])
    t_err = translation_direction_error_deg(rel_gt[:, :3, 3], rel_pred[:, :3, 3])

    return r_err.cpu().numpy(), t_err.cpu().numpy()


def calculate_auc_np(r_error: np.ndarray, t_error: np.ndarray, max_threshold: int) -> float:
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    if len(max_errors) == 0:
        return float("nan")
    normalized_hist = histogram.astype(np.float64) / float(len(max_errors))
    return float(np.mean(np.cumsum(normalized_hist)))


def summarize_errors(r_error: np.ndarray, t_error: np.ndarray) -> dict:
    if r_error.size == 0 or t_error.size == 0:
        return {
            "num_pairs": 0,
            "rot_mean_deg": float("nan"),
            "rot_median_deg": float("nan"),
            "trans_mean_deg": float("nan"),
            "trans_median_deg": float("nan"),
            "R_ACC@5": float("nan"),
            "T_ACC@5": float("nan"),
            "RT_ACC@5": float("nan"),
            "AUC@30": float("nan"),
            "AUC@15": float("nan"),
            "AUC@5": float("nan"),
            "AUC@3": float("nan"),
        }

    return {
        "num_pairs": int(r_error.size),
        "rot_mean_deg": float(np.mean(r_error)),
        "rot_median_deg": float(np.median(r_error)),
        "trans_mean_deg": float(np.mean(t_error)),
        "trans_median_deg": float(np.median(t_error)),
        "R_ACC@5": float(np.mean(r_error < 5.0)),
        "T_ACC@5": float(np.mean(t_error < 5.0)),
        "RT_ACC@5": float(np.mean(np.maximum(r_error, t_error) < 5.0)),
        "AUC@30": calculate_auc_np(r_error, t_error, max_threshold=30),
        "AUC@15": calculate_auc_np(r_error, t_error, max_threshold=15),
        "AUC@5": calculate_auc_np(r_error, t_error, max_threshold=5),
        "AUC@3": calculate_auc_np(r_error, t_error, max_threshold=3),
    }


def evaluate_sequence(
    model,
    scene_name: str,
    sequence_dir: Path,
    num_frames: int,
    seed: int,
    save_pair_errors: bool,
) -> Optional[tuple[dict, np.ndarray, np.ndarray]]:
    records = collect_frame_records(sequence_dir)
    if len(records) < 2:
        logger.warning("Skipping %s/%s: need at least 2 valid frames, got %d.", scene_name, sequence_dir.name, len(records))
        return None

    sampled = sample_records(records, scene_name, sequence_dir.name, num_frames, seed)
    if len(sampled) < 2:
        logger.warning("Skipping %s/%s: sampled fewer than 2 frames.", scene_name, sequence_dir.name)
        return None

    image_paths = [item["color_path"] for item in sampled]
    pose_paths = [item["pose_path"] for item in sampled]
    frame_ids = [item["frame_id"] for item in sampled]

    predictions = predict(image_paths, model)
    pred_w2c = decode_pred_w2c(predictions)
    gt_w2c = load_gt_w2c(pose_paths)

    if pred_w2c.shape[0] != gt_w2c.shape[0]:
        raise RuntimeError(
            f"Frame count mismatch in {scene_name}/{sequence_dir.name}: "
            f"pred={pred_w2c.shape[0]}, gt={gt_w2c.shape[0]}"
        )

    r_error, t_error = se3_to_relative_pose_error(pred_w2c, gt_w2c)
    metrics = summarize_errors(r_error, t_error)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sequence_result = {
        "scene": scene_name,
        "sequence": sequence_dir.name,
        "num_frames": len(frame_ids),
        "frame_ids": frame_ids,
        "metrics": metrics,
    }

    if save_pair_errors:
        sequence_result["r_error_deg"] = r_error.tolist()
        sequence_result["t_error_deg"] = t_error.tolist()

    return sequence_result, r_error, t_error


def format_metrics_for_log(metrics: dict) -> str:
    return (
        f"pairs={metrics['num_pairs']}, "
        f"R_ACC@5={metrics['R_ACC@5']:.4f}, "
        f"T_ACC@5={metrics['T_ACC@5']:.4f}, "
        f"RT_ACC@5={metrics['RT_ACC@5']:.4f}, "
        f"AUC@30={metrics['AUC@30']:.4f}, "
        f"AUC@15={metrics['AUC@15']:.4f}, "
        f"AUC@5={metrics['AUC@5']:.4f}, "
        f"AUC@3={metrics['AUC@3']:.4f}"
    )


def main() -> None:
    configure_logging()
    args = parse_args()

    if args.num_frames == 1:
        raise ValueError("num_frames=1 cannot form pose pairs. Please use 2 or more, or <=0 for all frames.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scenes = discover_scenes(args.data_root, args.scenes)
    if not scenes:
        raise ValueError("No valid scenes found to evaluate.")

    logger.info("Loading VGGT model from %s", args.model_path)
    # Pose-only evaluation: keep camera head enabled and disable unrelated heads.
    model = load_model(
        args.model_path,
        model_args={"enable_point": False, "enable_depth": False, "enable_track": False},
        backend=args.backend,
        quant_config={
            "smoothquant_path": args.smoothquant_scale_path,
            "smoothquant_strict": not args.smoothquant_allow_missing,
        },
    )

    all_scene_results: dict[str, dict] = {}
    all_r_errors: list[np.ndarray] = []
    all_t_errors: list[np.ndarray] = []

    for scene_name in scenes:
        scene_dir = args.data_root / scene_name
        sequence_dirs = discover_sequences(scene_dir, args.split)
        if not sequence_dirs:
            logger.warning("No valid sequences found in scene %s for split '%s'.", scene_name, args.split)
            continue

        logger.info("Evaluating scene %s with %d sequence(s)", scene_name, len(sequence_dirs))

        scene_sequence_results = []
        scene_r_errors = []
        scene_t_errors = []

        for seq_dir in sequence_dirs:
            eval_out = evaluate_sequence(
                model=model,
                scene_name=scene_name,
                sequence_dir=seq_dir,
                num_frames=args.num_frames,
                seed=args.seed,
                save_pair_errors=args.save_pair_errors,
            )
            if eval_out is None:
                continue

            result, r_err, t_err = eval_out

            scene_sequence_results.append(result)
            if r_err.size > 0 and t_err.size > 0:
                scene_r_errors.append(r_err)
                scene_t_errors.append(t_err)

            logger.info(
                "[%s/%s] %s",
                scene_name,
                seq_dir.name,
                format_metrics_for_log(result["metrics"]),
            )

        if not scene_sequence_results:
            logger.warning("No valid evaluation result in scene %s.", scene_name)
            continue

        if scene_r_errors and scene_t_errors:
            scene_r_all = np.concatenate(scene_r_errors, axis=0)
            scene_t_all = np.concatenate(scene_t_errors, axis=0)
            scene_metrics = summarize_errors(scene_r_all, scene_t_all)

            all_r_errors.append(scene_r_all)
            all_t_errors.append(scene_t_all)
        else:
            scene_metrics = summarize_errors(np.empty((0,)), np.empty((0,)))

        logger.info("[scene=%s] %s", scene_name, format_metrics_for_log(scene_metrics))

        all_scene_results[scene_name] = {
            "num_sequences": len(scene_sequence_results),
            "metrics": scene_metrics,
            "sequences": scene_sequence_results,
        }

    if not all_scene_results:
        raise RuntimeError("Evaluation finished but no valid sequence was processed.")

    if all_r_errors and all_t_errors:
        overall_r = np.concatenate(all_r_errors, axis=0)
        overall_t = np.concatenate(all_t_errors, axis=0)
        overall_metrics = summarize_errors(overall_r, overall_t)
    else:
        overall_metrics = summarize_errors(np.empty((0,)), np.empty((0,)))

    logger.info("[overall] %s", format_metrics_for_log(overall_metrics))

    payload = {
        "config": {
            "data_root": str(args.data_root),
            "model_path": str(args.model_path),
            "split": args.split,
            "scenes": scenes,
            "num_frames": args.num_frames,
            "seed": args.seed,
            "smoothquant_scale_path": str(args.smoothquant_scale_path) if args.smoothquant_scale_path is not None else None,
            "smoothquant_allow_missing": args.smoothquant_allow_missing,
        },
        "overall": overall_metrics,
        "scenes": all_scene_results,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Saved detailed results to %s", args.output_json)


if __name__ == "__main__":
    main()