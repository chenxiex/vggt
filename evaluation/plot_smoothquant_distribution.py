import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, PercentFormatter


DEFAULT_FIFTH_SIZE_PT = 10.5
TARGET_LAYER_PREFIXES = ("aggregator.frame_blocks", "aggregator.global_blocks")
SCALE_DESIGN_PER_BLOCK = "per_block"
SCALE_DESIGN_PER_PROJECTION = "per_projection"
DEFAULT_CURVE_METRICS = [
    "abs_p95_ratio_act_over_weight",
    "abs_p99_ratio_act_over_weight",
    "abs_p99.9_ratio_act_over_weight",
    "abs_max_ratio_act_over_weight",
]


def percentile_key(percentile: float) -> str:
    if float(percentile).is_integer():
        return str(int(percentile))
    return f"{percentile}".rstrip("0").rstrip(".")


def safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def is_target_layer(layer_name: str) -> bool:
    return layer_name.startswith(TARGET_LAYER_PREFIXES)


def estimate_abs_percentile_from_hist(
    percentile: float,
    hist_counts: np.ndarray,
    underflow_count: int,
    overflow_count: int,
    log_edges: np.ndarray,
    zero_eps: float,
    abs_max: float,
) -> float:
    if percentile <= 0.0:
        return 0.0
    if percentile >= 100.0:
        return float(abs_max)

    total = int(underflow_count + overflow_count + int(hist_counts.sum()))
    if total <= 0:
        return 0.0

    target = (percentile / 100.0) * (total - 1)
    cumulative = float(underflow_count)
    if target < cumulative:
        return 0.0

    for idx in range(hist_counts.size):
        bin_count = int(hist_counts[idx])
        if bin_count <= 0:
            continue

        next_cumulative = cumulative + bin_count
        if target < next_cumulative:
            left = float(log_edges[idx])
            right = float(log_edges[idx + 1])
            frac = (target - cumulative) / max(bin_count, 1)
            frac = min(max(frac, 0.0), 1.0)
            log_value = left + frac * (right - left)
            value = (10.0 ** log_value) - zero_eps
            return float(max(value, 0.0))
        cumulative = next_cumulative

    return float(abs_max)


def rebuild_global_stats_from_per_layer(
    per_layer: Dict[str, Dict[str, Any]],
    percentiles: Sequence[float],
    zero_eps: float,
) -> Dict[str, Any]:
    total_count = 0.0
    total_sum = 0.0
    total_sum_sq = 0.0
    total_abs_sum = 0.0
    total_zero = 0.0
    total_pos = 0.0
    total_neg = 0.0

    min_value: float | None = None
    max_value: float | None = None
    abs_max = 0.0

    hist_edges: np.ndarray | None = None
    hist_counts: np.ndarray | None = None
    underflow_count = 0
    overflow_count = 0

    for _, layer_stats in per_layer.items():
        count = float(layer_stats.get("count", 0.0))
        if count <= 0.0:
            continue

        mean = float(layer_stats["mean"])
        std = float(layer_stats["std"])
        abs_mean = float(layer_stats["abs_mean"])

        total_count += count
        total_sum += count * mean
        total_sum_sq += count * (std * std + mean * mean)
        total_abs_sum += count * abs_mean
        total_zero += count * float(layer_stats["zero_ratio"])
        total_pos += count * float(layer_stats["positive_ratio"])
        total_neg += count * float(layer_stats["negative_ratio"])

        layer_min = float(layer_stats["min"])
        layer_max = float(layer_stats["max"])
        layer_abs_max = float(layer_stats["abs_max"])
        min_value = layer_min if min_value is None else min(min_value, layer_min)
        max_value = layer_max if max_value is None else max(max_value, layer_max)
        abs_max = max(abs_max, layer_abs_max)

        hist = layer_stats["abs_log10_histogram"]
        cur_edges = np.asarray(hist["bin_edges"], dtype=np.float64)
        cur_counts = np.asarray(hist["counts"], dtype=np.int64)

        if hist_edges is None:
            hist_edges = cur_edges
            hist_counts = np.zeros_like(cur_counts, dtype=np.int64)
        else:
            if cur_edges.shape != hist_edges.shape or not np.allclose(cur_edges, hist_edges):
                raise ValueError("Inconsistent histogram bin edges across layers")

        assert hist_counts is not None
        hist_counts += cur_counts
        underflow_count += int(hist.get("underflow_count", 0))
        overflow_count += int(hist.get("overflow_count", 0))

    if total_count <= 0.0 or hist_edges is None or hist_counts is None:
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
            "abs_percentiles": {percentile_key(p): 0.0 for p in percentiles},
            "abs_log10_histogram": {
                "bin_edges": [],
                "counts": [],
                "underflow_count": 0,
                "overflow_count": 0,
            },
        }

    mean = total_sum / total_count
    variance = max(total_sum_sq / total_count - mean * mean, 0.0)

    abs_percentiles = {
        percentile_key(p): estimate_abs_percentile_from_hist(
            percentile=float(p),
            hist_counts=hist_counts,
            underflow_count=underflow_count,
            overflow_count=overflow_count,
            log_edges=hist_edges,
            zero_eps=zero_eps,
            abs_max=abs_max,
        )
        for p in percentiles
    }

    return {
        "count": int(total_count),
        "min": float(min_value if min_value is not None else 0.0),
        "max": float(max_value if max_value is not None else 0.0),
        "mean": float(mean),
        "std": float(np.sqrt(variance)),
        "abs_mean": float(total_abs_sum / total_count),
        "abs_max": float(abs_max),
        "zero_ratio": float(total_zero / total_count),
        "positive_ratio": float(total_pos / total_count),
        "negative_ratio": float(total_neg / total_count),
        "abs_percentiles": abs_percentiles,
        "abs_log10_histogram": {
            "bin_edges": [float(v) for v in hist_edges.tolist()],
            "counts": [int(v) for v in hist_counts.tolist()],
            "underflow_count": int(underflow_count),
            "overflow_count": int(overflow_count),
        },
    }


def restrict_stats_to_frame_global(stats: Dict[str, Any]) -> Dict[str, Any]:
    scoped = copy.deepcopy(stats)

    for section in ("weights", "activations"):
        per_layer = scoped[section].get("per_layer", {})
        scoped[section]["per_layer"] = {
            layer_name: layer_stats
            for layer_name, layer_stats in per_layer.items()
            if is_target_layer(layer_name)
        }

    update_counts = scoped.get("activations", {}).get("update_counts", {})
    scoped["activations"]["update_counts"] = {
        layer_name: count for layer_name, count in update_counts.items() if is_target_layer(layer_name)
    }

    comp_per_layer = scoped.get("comparison", {}).get("per_layer", {})
    scoped["comparison"]["per_layer"] = {
        layer_name: layer_comp for layer_name, layer_comp in comp_per_layer.items() if is_target_layer(layer_name)
    }

    meta = scoped.setdefault("meta", {})
    percentiles = meta.get("percentiles", [50.0, 90.0, 95.0, 99.0, 99.9])
    percentiles = [float(v) for v in percentiles]
    zero_eps = float(meta.get("zero_eps", 1e-8))

    scoped["weights"]["global"] = rebuild_global_stats_from_per_layer(
        scoped["weights"]["per_layer"], percentiles=percentiles, zero_eps=zero_eps
    )
    scoped["activations"]["global"] = rebuild_global_stats_from_per_layer(
        scoped["activations"]["per_layer"], percentiles=percentiles, zero_eps=zero_eps
    )

    weight_global = scoped["weights"]["global"]
    act_global = scoped["activations"]["global"]
    percentile_keys = [percentile_key(p) for p in percentiles]
    global_comp: Dict[str, float | None] = {
        "abs_max_ratio_act_over_weight": safe_divide(float(act_global["abs_max"]), float(weight_global["abs_max"]))
    }
    for key in percentile_keys:
        act_p = float(act_global["abs_percentiles"][key])
        weight_p = float(weight_global["abs_percentiles"][key])
        global_comp[f"abs_p{key}_ratio_act_over_weight"] = safe_divide(act_p, weight_p)
    scoped["comparison"]["global"] = global_comp

    selected_layers = len(scoped["weights"]["per_layer"])
    meta["layer_scope"] = "frame_global_only"
    meta["layer_name_prefixes"] = list(TARGET_LAYER_PREFIXES)
    meta["num_layers"] = int(selected_layers)
    meta["num_layers_selected"] = int(selected_layers)

    return scoped


def infer_point_granularity(stats: Dict[str, Any]) -> str:
    meta = stats.get("meta", {})
    point_granularity = str(meta.get("point_granularity", "")).strip().lower()
    if point_granularity in {"projection", "block"}:
        return point_granularity

    smooth_scale_design = str(meta.get("smooth_scale_design", "")).strip().lower()
    if smooth_scale_design == SCALE_DESIGN_PER_PROJECTION:
        return "projection"
    if smooth_scale_design == SCALE_DESIGN_PER_BLOCK:
        return "block"

    layer_names = list(stats.get("comparison", {}).get("per_layer", {}).keys())
    has_qkv_or_proj = any(layer_name.endswith(".qkv") or layer_name.endswith(".proj") for layer_name in layer_names)
    return "projection" if has_qkv_or_proj else "block"


def metric_to_unit_key(layer_name: str, point_granularity: str) -> str:
    if point_granularity != "block":
        return layer_name

    suffix = layer_name.rsplit(".", 1)[-1]
    if suffix in {"qkv", "proj"}:
        return layer_name.rsplit(".", 1)[0]
    return layer_name


def ascii_float_formatter(value: float, _pos: int) -> str:
    text = f"{value:.3g}"
    return text


def ascii_sci_formatter(value: float, _pos: int) -> str:
    if value <= 0:
        return "0"
    text = f"{value:.0e}"
    mantissa, exponent = text.split("e")
    exp_int = int(exponent)
    return f"{mantissa}e{exp_int}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot publication-ready SmoothQuant distribution figures from profile_smoothquant_distribution output"
    )
    parser.add_argument(
        "--stats_path",
        type=Path,
        required=True,
        help="Path to distribution stats generated by profile_smoothquant_distribution (.json or .pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/smoothquant_figures"),
        help="Directory for output figures",
    )
    parser.add_argument(
        "--font_path",
        type=str,
        default="evaluation/simsun.ttc",
        help="Path or filename of SimSun font collection",
    )
    parser.add_argument(
        "--font_size",
        type=float,
        default=DEFAULT_FIFTH_SIZE_PT,
        help="Base font size in points. Fifth size is 10.5 pt by default.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Raster figure DPI")
    parser.add_argument(
        "--formats",
        type=str,
        default="png,pdf",
        help="Comma-separated output formats. Supported: png,pdf,svg",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Top-K layers in the per-layer ratio bar chart",
    )
    parser.add_argument(
        "--ratio_metric",
        type=str,
        default="abs_p99_ratio_act_over_weight",
        help="Metric key used in top-k bar chart",
    )
    parser.add_argument(
        "--curve_metrics",
        type=str,
        default=",".join(DEFAULT_CURVE_METRICS),
        help="Comma-separated metric keys for rank-curve figure",
    )
    return parser.parse_args()


def parse_csv_list(text: str) -> List[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("No valid entries parsed from CSV string")
    return values


def parse_formats(text: str) -> List[str]:
    formats = parse_csv_list(text)
    allowed = {"png", "pdf", "svg"}
    bad = [fmt for fmt in formats if fmt not in allowed]
    if bad:
        raise ValueError(f"Unsupported formats: {bad}. Allowed: {sorted(allowed)}")
    return formats


def resolve_font_path(font_path: str) -> Path:
    path = Path(font_path)
    if path.exists():
        return path

    fallback_local = Path("evaluation") / path.name
    if fallback_local.exists():
        return fallback_local

    # If only a filename is passed, try to locate it among system fonts.
    if path.parent == Path("."):
        target = path.name.lower()
        for font_file in font_manager.findSystemFonts(fontpaths=None, fontext="ttf"):
            if Path(font_file).name.lower() == target:
                return Path(font_file)
        # Some systems register TTC files but do not list them via fontext="ttf".
        for font_file in font_manager.findSystemFonts(fontpaths=None, fontext="afm"):
            if Path(font_file).name.lower() == target:
                return Path(font_file)

    raise FileNotFoundError(
        f"Font file not found: {font_path}. Please provide --font_path /path/to/simsun.ttc"
    )


def configure_plot_style(font_file: Path, font_size: float) -> str:
    font_manager.fontManager.addfont(str(font_file))
    font_name = font_manager.FontProperties(fname=str(font_file)).get_name()

    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": float(font_size),
            "axes.titlesize": float(font_size),
            "axes.labelsize": float(font_size),
            "legend.fontsize": float(font_size) - 0.5,
            "xtick.labelsize": float(font_size) - 0.5,
            "ytick.labelsize": float(font_size) - 0.5,
            "axes.unicode_minus": False,
            "figure.dpi": 100,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )
    return font_name


def load_stats(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")

    if path.suffix.lower() == ".pt":
        data = torch.load(path, map_location="cpu")
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError("Stats file should contain a dictionary")

    for required in ("weights", "activations", "comparison"):
        if required not in data:
            raise KeyError(f"Missing required top-level key: {required}")

    return data


def histogram_arrays(kind_stats: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    hist = kind_stats["global"]["abs_log10_histogram"]
    edges = np.asarray(hist["bin_edges"], dtype=np.float64)
    counts = np.asarray(hist["counts"], dtype=np.float64)

    if edges.ndim != 1 or counts.ndim != 1 or len(edges) != len(counts) + 1:
        raise ValueError("Invalid histogram shape in stats file")

    underflow = float(hist.get("underflow_count", 0.0))
    overflow = float(hist.get("overflow_count", 0.0))

    total = float(underflow + overflow + counts.sum())
    if total <= 0.0:
        total = 1.0

    centers = 0.5 * (edges[:-1] + edges[1:])
    density = counts / total
    cdf = (underflow + np.cumsum(counts)) / total
    return centers, density, cdf, float(edges[1]), float(edges[-1])


def metric_values(
    comparison_per_layer: Dict[str, Dict[str, Any]],
    metric: str,
    point_granularity: str,
) -> List[Tuple[str, float]]:
    values_map: Dict[str, List[float]] = {}
    for layer_name, layer_data in comparison_per_layer.items():
        value = layer_data.get(metric)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value) and value > 0.0:
            unit_key = metric_to_unit_key(layer_name, point_granularity)
            values_map.setdefault(unit_key, []).append(value)

    values: List[Tuple[str, float]] = []
    for unit_key, unit_values in values_map.items():
        values.append((unit_key, float(np.mean(unit_values))))
    return values


def save_figure(fig: plt.Figure, out_base: Path, formats: Sequence[str], dpi: int) -> List[Path]:
    saved: List[Path] = []
    for fmt in formats:
        out_path = out_base.with_suffix(f".{fmt}")
        if fmt in {"png"}:
            fig.savefig(out_path, dpi=dpi)
        else:
            fig.savefig(out_path)
        saved.append(out_path)
    plt.close(fig)
    return saved


def plot_global_histogram(
    stats: Dict[str, Any],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    w_center, w_density, _, _, _ = histogram_arrays(stats["weights"])
    a_center, a_density, _, _, _ = histogram_arrays(stats["activations"])

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(w_center, w_density, color="#1f77b4", linewidth=1.8, label="Weight")
    ax.plot(a_center, a_density, color="#d62728", linewidth=1.8, label="Activation")

    pctl_w = stats["weights"]["global"]["abs_percentiles"]
    pctl_a = stats["activations"]["global"]["abs_percentiles"]

    if "99" in pctl_w and pctl_w["99"] > 0:
        ax.axvline(np.log10(float(pctl_w["99"])), color="#1f77b4", linestyle="--", linewidth=1.0, alpha=0.55)
    if "99" in pctl_a and pctl_a["99"] > 0:
        ax.axvline(np.log10(float(pctl_a["99"])), color="#d62728", linestyle="--", linewidth=1.0, alpha=0.55)

    ax.set_title("Global Magnitude Distribution")
    ax.set_xlabel("log10(|x|)")
    ax.set_ylabel("Frequency")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(FuncFormatter(ascii_float_formatter))
    ax.legend(loc="best", frameon=True)

    return save_figure(fig, output_dir / "fig1_global_logabs_hist", formats, dpi)


def plot_global_cdf(
    stats: Dict[str, Any],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    _, _, w_cdf, w_xmin, w_xmax = histogram_arrays(stats["weights"])
    _, _, a_cdf, a_xmin, a_xmax = histogram_arrays(stats["activations"])

    hist_w = stats["weights"]["global"]["abs_log10_histogram"]
    hist_a = stats["activations"]["global"]["abs_log10_histogram"]

    w_edges = np.asarray(hist_w["bin_edges"], dtype=np.float64)
    a_edges = np.asarray(hist_a["bin_edges"], dtype=np.float64)

    w_x = np.power(10.0, w_edges[1:])
    a_x = np.power(10.0, a_edges[1:])

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(w_x, w_cdf, color="#1f77b4", linewidth=1.8, label="Weight")
    ax.plot(a_x, a_cdf, color="#d62728", linewidth=1.8, label="Activation")

    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(np.power(10.0, min(w_xmin, a_xmin)), np.power(10.0, max(w_xmax, a_xmax)))
    ax.set_title("Global CDF of |x|")
    ax.set_xlabel("|x| (log scale)")
    ax.set_ylabel("CDF")
    ax.xaxis.set_major_formatter(FuncFormatter(ascii_sci_formatter))
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.9, alpha=0.5)
    ax.axhline(0.99, color="gray", linestyle=":", linewidth=0.9, alpha=0.5)
    ax.legend(loc="lower right", frameon=True)

    return save_figure(fig, output_dir / "fig2_global_abs_cdf", formats, dpi)


def plot_ratio_rank_curves(
    stats: Dict[str, Any],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    metrics: Sequence[str],
    point_granularity: str,
) -> List[Path]:
    per_layer = stats["comparison"]["per_layer"]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]

    plotted = 0
    for idx, metric in enumerate(metrics):
        vals = [v for _, v in metric_values(per_layer, metric, point_granularity)]
        if not vals:
            continue

        vals = sorted(vals, reverse=True)
        ranks = np.arange(1, len(vals) + 1, dtype=np.int64)
        color = colors[idx % len(colors)]
        ax.plot(ranks, vals, linewidth=1.6, color=color, label=metric)
        plotted += 1

    if plotted == 0:
        raise ValueError("None of the requested curve metrics are available in comparison.per_layer")

    unit_label = "Projection" if point_granularity == "projection" else "Block"

    ax.set_yscale("log")
    ax.set_xlabel(f"{unit_label} Rank (sorted by ratio)")
    ax.set_ylabel("Activation / Weight")
    ax.set_title(f"{unit_label}-wise Ratio Curves")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.yaxis.set_major_formatter(FuncFormatter(ascii_sci_formatter))
    ax.legend(loc="best", frameon=True)

    return save_figure(fig, output_dir / "fig3_layer_ratio_rank_curves", formats, dpi)


def shorten_layer_name(name: str, max_len: int = 46) -> str:
    if len(name) <= max_len:
        return name
    return "..." + name[-(max_len - 3) :]


def plot_topk_ratio_bar(
    stats: Dict[str, Any],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    metric: str,
    topk: int,
    point_granularity: str,
) -> List[Path]:
    values = metric_values(stats["comparison"]["per_layer"], metric, point_granularity)
    if not values:
        raise ValueError(f"Metric not found or empty in comparison.per_layer: {metric}")

    values = sorted(values, key=lambda x: x[1], reverse=True)
    if topk > 0:
        values = values[:topk]

    labels = [shorten_layer_name(name) for name, _ in values][::-1]
    score = [val for _, val in values][::-1]

    fig_h = max(4.2, 0.34 * len(values) + 1.4)
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    y = np.arange(len(values), dtype=np.int64)

    ax.barh(y, score, color="#4e79a7", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Activation / Weight")
    unit_label = "Projections" if point_granularity == "projection" else "Blocks"
    ax.set_title(f"Top-{len(values)} {unit_label} by {metric}")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)

    return save_figure(fig, output_dir / "fig4_topk_layer_ratio", formats, dpi)


def main() -> None:
    args = parse_args()

    if args.topk <= 0:
        raise ValueError("topk must be positive")
    if args.font_size <= 0.0:
        raise ValueError("font_size must be positive")
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")

    formats = parse_formats(args.formats)
    curve_metrics = parse_csv_list(args.curve_metrics)

    stats = load_stats(args.stats_path)
    stats = restrict_stats_to_frame_global(stats)
    point_granularity = infer_point_granularity(stats)

    font_file = resolve_font_path(args.font_path)
    font_name = configure_plot_style(font_file=font_file, font_size=args.font_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    saved_paths.extend(plot_global_histogram(stats, args.output_dir, formats, args.dpi))
    saved_paths.extend(plot_global_cdf(stats, args.output_dir, formats, args.dpi))
    saved_paths.extend(
        plot_ratio_rank_curves(
            stats,
            args.output_dir,
            formats,
            args.dpi,
            curve_metrics,
            point_granularity=point_granularity,
        )
    )
    saved_paths.extend(
        plot_topk_ratio_bar(
            stats,
            args.output_dir,
            formats,
            args.dpi,
            metric=args.ratio_metric,
            topk=args.topk,
            point_granularity=point_granularity,
        )
    )

    print(f"Using font: {font_name} ({font_file})")
    print(f"Layer scope: frame/global only ({stats['meta'].get('num_layers_selected', stats['meta'].get('num_layers'))} points)")
    print(f"Point granularity: {point_granularity}")
    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
