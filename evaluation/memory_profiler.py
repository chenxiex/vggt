import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from vggt.models.vggt import VGGT


VGGT_ENABLE_CUDA_MEM_STATS_ENV = "VGGT_ENABLE_CUDA_MEM_STATS"
VGGT_CUDA_MEM_STATS_FILE_ENV = "VGGT_CUDA_MEM_STATS_FILE"
VGGT_CUDA_MEM_STATS_LEVEL_ENV = "VGGT_CUDA_MEM_STATS_LEVEL"

_REPORT_LEVEL_ALIASES = {
    "0": "quiet",
    "off": "quiet",
    "none": "quiet",
    "quiet": "quiet",
    "1": "summary",
    "basic": "summary",
    "summary": "summary",
    "2": "detailed",
    "detail": "detailed",
    "detailed": "detailed",
    "3": "plot",
    "chart": "plot",
    "plot": "plot",
    "visual": "plot",
}


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_report_level(name: str) -> str:
    value = os.environ.get(name, "summary")
    return _REPORT_LEVEL_ALIASES.get(value.strip().lower(), "summary")


def _bytes_to_mib(num_bytes: int) -> float:
    return round(num_bytes / (1024 ** 2), 4)


def _format_mib(value_mib: Optional[float]) -> str:
    if value_mib is None:
        return "n/a"
    return f"{value_mib:.2f} MiB"


def _tensor_num_bytes(tensor: torch.Tensor) -> int:
    return tensor.nelement() * tensor.element_size()


def _cuda_current_memory(device: torch.device) -> Dict[str, Any]:
    allocated_bytes = torch.cuda.memory_allocated(device)
    reserved_bytes = torch.cuda.memory_reserved(device)
    return {
        "allocated_bytes": allocated_bytes,
        "allocated_mib": _bytes_to_mib(allocated_bytes),
        "reserved_bytes": reserved_bytes,
        "reserved_mib": _bytes_to_mib(reserved_bytes),
    }


def _cuda_peak_memory(device: torch.device) -> Dict[str, Any]:
    peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
    peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
    return {
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_allocated_mib": _bytes_to_mib(peak_allocated_bytes),
        "peak_reserved_bytes": peak_reserved_bytes,
        "peak_reserved_mib": _bytes_to_mib(peak_reserved_bytes),
    }


class PredictionMemoryProfiler:
    def __init__(self, model: VGGT, device: torch.device, dtype: torch.dtype):
        self.enabled = _env_flag(VGGT_ENABLE_CUDA_MEM_STATS_ENV)
        self.output_path = os.environ.get(VGGT_CUDA_MEM_STATS_FILE_ENV)
        self.report_level = _env_report_level(VGGT_CUDA_MEM_STATS_LEVEL_ENV)
        self.active = self.enabled and torch.cuda.is_available() and device.type == "cuda"
        self._model = model
        self._device = device
        self._dtype = dtype
        self._handles: List[Any] = []
        self._observed_allocated_values: List[int] = []
        self._observed_reserved_values: List[int] = []
        self.stats: Dict[str, Any] = {
            "enabled": self.enabled,
            "active": self.active,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "autocast_dtype": str(dtype),
            "env": {
                "enable": VGGT_ENABLE_CUDA_MEM_STATS_ENV,
                "output_file": VGGT_CUDA_MEM_STATS_FILE_ENV,
                "report_level": VGGT_CUDA_MEM_STATS_LEVEL_ENV,
                "output_path": self.output_path,
                "resolved_report_level": self.report_level,
            },
            "reason": None if self.active else self._inactive_reason(),
            "model": self._model_metadata(model),
            "inputs": {},
            "snapshots": {},
            "modules": {},
            "overall_peak": None,
            "artifacts": {},
            "error": None,
        }

    def _inactive_reason(self) -> Optional[str]:
        if not self.enabled:
            return "disabled_by_env"
        if not torch.cuda.is_available():
            return "cuda_unavailable"
        if self._device.type != "cuda":
            return "non_cuda_device"
        return None

    def _model_metadata(self, model: VGGT) -> Dict[str, Any]:
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        parameter_bytes = sum(_tensor_num_bytes(parameter) for parameter in model.parameters())
        buffer_bytes = sum(_tensor_num_bytes(buffer) for buffer in model.buffers())
        return {
            "parameter_count": parameter_count,
            "parameter_bytes": parameter_bytes,
            "parameter_mib": _bytes_to_mib(parameter_bytes),
            "buffer_bytes": buffer_bytes,
            "buffer_mib": _bytes_to_mib(buffer_bytes),
        }

    def _synchronize(self) -> None:
        if self.active:
            torch.cuda.synchronize(self._device)

    def _remember_current_memory(self, snapshot: Dict[str, Any]) -> None:
        self._observed_allocated_values.append(snapshot["allocated_bytes"])
        self._observed_reserved_values.append(snapshot["reserved_bytes"])

    def _remember_peak_memory(self, peak: Dict[str, Any]) -> None:
        self._observed_allocated_values.append(peak["peak_allocated_bytes"])
        self._observed_reserved_values.append(peak["peak_reserved_bytes"])

    def record_snapshot(self, label: str) -> None:
        if not self.active:
            return

        self._synchronize()
        snapshot = _cuda_current_memory(self._device)
        self.stats["snapshots"][label] = snapshot
        self._remember_current_memory(snapshot)

    def record_input_metadata(self, images: torch.Tensor, image_paths: List[Path]) -> None:
        self.stats["inputs"] = {
            "num_images": len(image_paths),
            "paths": [str(path) for path in image_paths],
            "shape": list(images.shape),
            "dtype": str(images.dtype),
            "device": str(images.device),
            "tensor_bytes": _tensor_num_bytes(images),
            "tensor_mib": _bytes_to_mib(_tensor_num_bytes(images)),
        }

    def _register_peak_hook(self, module: torch.nn.Module, module_name: str, module_type: str, index: Optional[int] = None) -> None:
        if not self.active:
            return

        module_stats = self.stats["modules"].setdefault(
            module_name,
            {
                "name": module_name,
                "module_type": module_type,
                "index": index,
                "calls": 0,
                "before": None,
                "after": None,
                "peak": None,
                "peak_delta_allocated_bytes": None,
                "peak_delta_allocated_mib": None,
                "peak_delta_reserved_bytes": None,
                "peak_delta_reserved_mib": None,
            },
        )

        def pre_hook(_module, _inputs):
            self._synchronize()
            before = _cuda_current_memory(self._device)
            module_stats["calls"] += 1
            module_stats["before"] = before
            self._remember_current_memory(before)
            torch.cuda.reset_peak_memory_stats(self._device)

        def post_hook(_module, _inputs, _output):
            self._synchronize()
            after = _cuda_current_memory(self._device)
            peak = _cuda_peak_memory(self._device)
            before = module_stats["before"]

            module_stats["after"] = after
            module_stats["peak"] = peak
            module_stats["peak_delta_allocated_bytes"] = max(0, peak["peak_allocated_bytes"] - before["allocated_bytes"])
            module_stats["peak_delta_allocated_mib"] = _bytes_to_mib(module_stats["peak_delta_allocated_bytes"])
            module_stats["peak_delta_reserved_bytes"] = max(0, peak["peak_reserved_bytes"] - before["reserved_bytes"])
            module_stats["peak_delta_reserved_mib"] = _bytes_to_mib(module_stats["peak_delta_reserved_bytes"])

            self._remember_current_memory(after)
            self._remember_peak_memory(peak)

        self._handles.append(module.register_forward_pre_hook(pre_hook))
        self._handles.append(module.register_forward_hook(post_hook))

    def install_hooks(self) -> None:
        if not self.active:
            return

        aggregator = getattr(self._model, "aggregator", None)
        if aggregator is None:
            return

        patch_embed = getattr(aggregator, "patch_embed", None)
        if patch_embed is not None:
            self._register_peak_hook(patch_embed, "aggregator.patch_embed", "patch_embed")

        for index, block in enumerate(aggregator.frame_blocks):
            self._register_peak_hook(block, f"aggregator.frame_blocks.{index}", "frame_block", index=index)

        for index, block in enumerate(aggregator.global_blocks):
            self._register_peak_hook(block, f"aggregator.global_blocks.{index}", "global_block", index=index)

        for head_name in ("camera_head", "depth_head", "point_head", "track_head"):
            head = getattr(self._model, head_name, None)
            if head is not None:
                self._register_peak_hook(head, head_name, "head")

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def record_error(self, exc: BaseException) -> None:
        self.stats["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    def _selected_snapshots(self) -> List[Tuple[str, Dict[str, Any]]]:
        preferred = [
            "before_predict",
            "after_input_to_device",
            "before_model_forward",
            "after_model_forward",
        ]
        snapshots = self.stats["snapshots"]
        labels = [label for label in preferred if label in snapshots]
        labels.extend(label for label in snapshots if label not in labels)
        return [(label, snapshots[label]) for label in labels]

    def _grouped_modules(self) -> Dict[str, List[Dict[str, Any]]]:
        groups: Dict[str, List[Dict[str, Any]]] = {
            "patch_embed": [],
            "frame_block": [],
            "global_block": [],
            "head": [],
        }
        for module_stats in self.stats["modules"].values():
            module_type = module_stats["module_type"]
            groups.setdefault(module_type, []).append(module_stats)

        for module_type in groups:
            groups[module_type].sort(key=lambda item: (item["index"] is None, item["index"] if item["index"] is not None else item["name"]))
        return groups

    def _build_summary_lines(self) -> List[str]:
        lines = ["=== VGGT CUDA Memory Profile ==="]
        lines.append(f"status: {'active' if self.active else 'inactive'}")
        lines.append(f"report level: {self.report_level}")

        if not self.enabled:
            lines.append(f"profiling disabled by env {VGGT_ENABLE_CUDA_MEM_STATS_ENV}")
            return lines

        if not self.active:
            lines.append(f"profiling unavailable: {self.stats['reason']}")
            return lines

        model_stats = self.stats["model"]
        input_stats = self.stats["inputs"]
        overall_peak = self.stats["overall_peak"] or {}

        lines.append(f"device: {self.stats['device']} | autocast: {self.stats['autocast_dtype']}")
        lines.append(
            "model: "
            f"params={model_stats['parameter_count']:,} ({_format_mib(model_stats['parameter_mib'])}), "
            f"buffers={_format_mib(model_stats['buffer_mib'])}"
        )

        if input_stats:
            lines.append(
                "inputs: "
                f"num_images={input_stats['num_images']}, "
                f"shape={input_stats['shape']}, "
                f"dtype={input_stats['dtype']}, "
                f"tensor={_format_mib(input_stats['tensor_mib'])}"
            )

        if overall_peak:
            lines.append(
                "overall peak: "
                f"allocated={_format_mib(overall_peak['observed_peak_allocated_mib'])}, "
                f"reserved={_format_mib(overall_peak['observed_peak_reserved_mib'])}"
            )

        lines.append("snapshots:")
        for label, snapshot in self._selected_snapshots():
            lines.append(
                f"  - {label}: allocated={_format_mib(snapshot['allocated_mib'])}, reserved={_format_mib(snapshot['reserved_mib'])}"
            )

        if self.stats["error"]:
            lines.append(
                f"error: {self.stats['error']['type']}: {self.stats['error']['message']}"
            )

        return lines

    def _build_detailed_lines(self) -> List[str]:
        if not self.active:
            return []

        title_map = {
            "patch_embed": "patch embed",
            "frame_block": "frame blocks",
            "global_block": "global blocks",
            "head": "heads",
        }
        lines = ["module peaks:"]
        for module_type, modules in self._grouped_modules().items():
            if not modules:
                continue
            lines.append(f"  {title_map.get(module_type, module_type)}:")
            for module_stats in modules:
                lines.append(
                    "    - "
                    f"{module_stats['name']}: "
                    f"delta_allocated={_format_mib(module_stats['peak_delta_allocated_mib'])}, "
                    f"delta_reserved={_format_mib(module_stats['peak_delta_reserved_mib'])}, "
                    f"calls={module_stats['calls']}"
                )
        return lines

    def _report_lines(self) -> List[str]:
        lines = self._build_summary_lines()
        if self.report_level in {"detailed", "plot"}:
            lines.extend(self._build_detailed_lines())
        return lines

    def _default_artifact_stem(self) -> Path:
        if self.output_path:
            return Path(self.output_path).with_suffix("")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path.cwd() / f"vggt_cuda_mem_stats_{timestamp}"

    def _render_plot(self) -> Optional[Path]:
        if not self.active:
            return None

        module_peaks = [
            module_stats
            for module_stats in self.stats["modules"].values()
            if module_stats["peak_delta_allocated_mib"] is not None
        ]
        snapshots = self._selected_snapshots()
        if not module_peaks and not snapshots:
            return None

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            artifact_stem = self._default_artifact_stem()
            chart_path = artifact_stem.with_name(artifact_stem.name + "_chart").with_suffix(".png")
            chart_path.parent.mkdir(parents=True, exist_ok=True)

            figure, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)

            if snapshots:
                snapshot_labels = [label for label, _snapshot in snapshots]
                allocated = [snapshot["allocated_mib"] for _label, snapshot in snapshots]
                reserved = [snapshot["reserved_mib"] for _label, snapshot in snapshots]
                axes[0].plot(snapshot_labels, allocated, marker="o", label="allocated")
                axes[0].plot(snapshot_labels, reserved, marker="o", label="reserved")
                axes[0].set_title("Snapshot Memory Usage")
                axes[0].set_ylabel("MiB")
                axes[0].tick_params(axis="x", rotation=20)
                axes[0].legend()
            else:
                axes[0].set_axis_off()

            if module_peaks:
                module_peaks.sort(key=lambda item: item["peak_delta_allocated_mib"], reverse=True)
                labels = [module_stats["name"] for module_stats in module_peaks]
                values = [module_stats["peak_delta_allocated_mib"] for module_stats in module_peaks]
                axes[1].bar(labels, values)
                axes[1].set_title("Module Peak Delta Allocated Memory")
                axes[1].set_ylabel("MiB")
                axes[1].tick_params(axis="x", rotation=75)
            else:
                axes[1].set_axis_off()

            figure.savefig(chart_path, dpi=150)
            plt.close(figure)
            return chart_path
        except Exception as exc:
            self.stats["artifacts"]["chart_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            return None

    def _emit_report(self) -> None:
        if self.report_level == "quiet":
            return

        lines = self._report_lines()
        print("\n".join(lines), flush=True)

        if self.report_level == "plot":
            chart_path = self._render_plot()
            if chart_path is not None:
                self.stats["artifacts"]["chart"] = str(chart_path)
                print(f"chart: {chart_path}", flush=True)
            elif self.stats["artifacts"].get("chart_error"):
                chart_error = self.stats["artifacts"]["chart_error"]
                print(f"chart error: {chart_error['type']}: {chart_error['message']}", flush=True)

    def finalize(self) -> Dict[str, Any]:
        if self.active:
            overall_allocated_bytes = max(self._observed_allocated_values, default=0)
            overall_reserved_bytes = max(self._observed_reserved_values, default=0)
            self.stats["overall_peak"] = {
                "observed_peak_allocated_bytes": overall_allocated_bytes,
                "observed_peak_allocated_mib": _bytes_to_mib(overall_allocated_bytes),
                "observed_peak_reserved_bytes": overall_reserved_bytes,
                "observed_peak_reserved_mib": _bytes_to_mib(overall_reserved_bytes),
            }

        if self.enabled:
            self._emit_report()

        if self.enabled and self.output_path:
            payload = json.dumps(self.stats, indent=2)
            output_file = Path(self.output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(payload, encoding="utf-8")
            if self.report_level != "quiet":
                print(f"json: {self.output_path}", flush=True)

        return self.stats