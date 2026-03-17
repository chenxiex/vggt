import json
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import torch

from vggt.models.vggt import VGGT


VGGT_ENABLE_PROFILER_ENV = "VGGT_ENABLE_PROFILER"
VGGT_PROFILER_OUTPUT_DIR_ENV = "VGGT_PROFILER_OUTPUT_DIR"
VGGT_PROFILER_REPORT_LEVEL_ENV = "VGGT_PROFILER_REPORT_LEVEL"

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


def _format_ms(value_ms: Optional[float]) -> str:
    if value_ms is None:
        return "n/a"
    return f"{value_ms:.2f} ms"


def _shorten_timeline_label(label: str) -> str:
    label = label.replace("aggregator.frame_blocks.", "fb")
    label = label.replace("aggregator.global_blocks.", "gb")
    label = label.replace("aggregator.patch_embed", "pe")
    label = label.replace(":pre", "\u2193")
    label = label.replace(":post", "\u2191")
    return label


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
        self.enabled = _env_flag(VGGT_ENABLE_PROFILER_ENV)
        self.output_dir = os.environ.get(VGGT_PROFILER_OUTPUT_DIR_ENV)
        self.report_level = _env_report_level(VGGT_PROFILER_REPORT_LEVEL_ENV)
        self.active = self.enabled and torch.cuda.is_available() and device.type == "cuda"
        if self.output_dir:
            self._resolved_output_dir_path = Path(self.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._resolved_output_dir_path = Path.cwd() / f"vggt_profiler_{timestamp}"
        self._model = model
        self._device = device
        self._dtype = dtype
        self._handles: List[Any] = []
        self._started_at = perf_counter()
        self._observed_allocated_values: List[int] = []
        self._observed_reserved_values: List[int] = []
        self._snapshot_marks: List[Tuple[str, float]] = []
        self._timeline: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {
            "enabled": self.enabled,
            "active": self.active,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "autocast_dtype": str(dtype),
            "env": {
                "enable": VGGT_ENABLE_PROFILER_ENV,
                "output_dir": VGGT_PROFILER_OUTPUT_DIR_ENV,
                "report_level": VGGT_PROFILER_REPORT_LEVEL_ENV,
                "resolved_output_dir": str(self._resolved_output_dir_path),
                "resolved_report_level": self.report_level,
            },
            "reason": None if self.active else self._inactive_reason(),
            "model": self._model_metadata(model),
            "inputs": {},
            "timings": {
                "total_elapsed_ms": None,
                "snapshot_durations_ms": {},
                "model_forward_elapsed_ms": None,
                "module_total_elapsed_ms": None,
                "slowest_module": None,
            },
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

    def _record_timeline_entry(self, label: str, memory: Dict[str, Any]) -> None:
        if not self.active:
            return
        now = perf_counter()
        self._timeline.append({
            "seq": len(self._timeline),
            "label": label,
            "allocated_mib": memory["allocated_mib"],
            "reserved_mib": memory["reserved_mib"],
            "elapsed_ms_from_start": round((now - self._started_at) * 1000.0, 4),
        })

    def record_snapshot(self, label: str) -> None:
        if not self.active:
            return

        self._synchronize()
        now = perf_counter()
        snapshot = _cuda_current_memory(self._device)
        snapshot["elapsed_ms_from_start"] = round((now - self._started_at) * 1000.0, 4)
        self.stats["snapshots"][label] = snapshot
        self._snapshot_marks.append((label, now))
        self._remember_current_memory(snapshot)
        self._record_timeline_entry(label, snapshot)

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
                "total_elapsed_ms": 0.0,
                "avg_elapsed_ms": None,
                "max_elapsed_ms": None,
                "last_elapsed_ms": None,
                "_call_start_perf_counter_s": None,
            },
        )

        def pre_hook(_module, _inputs):
            self._synchronize()
            before = _cuda_current_memory(self._device)
            module_stats["_call_start_perf_counter_s"] = perf_counter()
            module_stats["calls"] += 1
            module_stats["before"] = before
            self._remember_current_memory(before)
            self._record_timeline_entry(f"{module_name}:pre", before)
            torch.cuda.reset_peak_memory_stats(self._device)

        def post_hook(_module, _inputs, _output):
            self._synchronize()
            end = perf_counter()
            after = _cuda_current_memory(self._device)
            peak = _cuda_peak_memory(self._device)
            before = module_stats["before"]
            start = module_stats.get("_call_start_perf_counter_s")
            elapsed_ms = None
            if isinstance(start, float):
                elapsed_ms = max(0.0, (end - start) * 1000.0)

            module_stats["after"] = after
            module_stats["peak"] = peak
            module_stats["peak_delta_allocated_bytes"] = max(0, peak["peak_allocated_bytes"] - before["allocated_bytes"])
            module_stats["peak_delta_allocated_mib"] = _bytes_to_mib(module_stats["peak_delta_allocated_bytes"])
            module_stats["peak_delta_reserved_bytes"] = max(0, peak["peak_reserved_bytes"] - before["reserved_bytes"])
            module_stats["peak_delta_reserved_mib"] = _bytes_to_mib(module_stats["peak_delta_reserved_bytes"])
            module_stats["last_elapsed_ms"] = elapsed_ms
            if elapsed_ms is not None:
                module_stats["total_elapsed_ms"] += elapsed_ms
                module_stats["avg_elapsed_ms"] = module_stats["total_elapsed_ms"] / max(module_stats["calls"], 1)
                if module_stats["max_elapsed_ms"] is None or elapsed_ms > module_stats["max_elapsed_ms"]:
                    module_stats["max_elapsed_ms"] = elapsed_ms

            self._remember_current_memory(after)
            self._remember_peak_memory(peak)
            self._record_timeline_entry(f"{module_name}:post", after)

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
        lines = ["=== VGGT CUDA Compute Profile ==="]
        lines.append(f"status: {'active' if self.active else 'inactive'}")
        lines.append(f"report level: {self.report_level}")

        if not self.enabled:
            lines.append(f"profiling disabled by env {VGGT_ENABLE_PROFILER_ENV}")
            return lines

        if not self.active:
            lines.append(f"profiling unavailable: {self.stats['reason']}")
            return lines

        model_stats = self.stats["model"]
        input_stats = self.stats["inputs"]
        overall_peak = self.stats["overall_peak"] or {}
        timings = self.stats["timings"]

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

        lines.append(
            "timing: "
            f"total={_format_ms(timings['total_elapsed_ms'])}, "
            f"model_forward={_format_ms(timings['model_forward_elapsed_ms'])}, "
            f"module_sum={_format_ms(timings['module_total_elapsed_ms'])}"
        )

        if timings["slowest_module"]:
            slowest = timings["slowest_module"]
            lines.append(
                "slowest module: "
                f"{slowest['name']} (avg={_format_ms(slowest['avg_elapsed_ms'])}, max={_format_ms(slowest['max_elapsed_ms'])})"
            )

        lines.append("snapshots:")
        for label, snapshot in self._selected_snapshots():
            lines.append(
                "  - "
                f"{label}: allocated={_format_mib(snapshot['allocated_mib'])}, "
                f"reserved={_format_mib(snapshot['reserved_mib'])}, "
                f"elapsed={_format_ms(snapshot.get('elapsed_ms_from_start'))}"
            )

        if timings["snapshot_durations_ms"]:
            lines.append("snapshot durations:")
            for edge, elapsed_ms in timings["snapshot_durations_ms"].items():
                lines.append(f"  - {edge}: {_format_ms(elapsed_ms)}")

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
                    f"avg_time={_format_ms(module_stats['avg_elapsed_ms'])}, "
                    f"max_time={_format_ms(module_stats['max_elapsed_ms'])}, "
                    f"calls={module_stats['calls']}"
                )
        return lines

    def _report_lines(self) -> List[str]:
        lines = self._build_summary_lines()
        if self.report_level in {"detailed", "plot"}:
            lines.extend(self._build_detailed_lines())
        return lines

    def _output_dir_path(self) -> Path:
        return self._resolved_output_dir_path

    def _json_output_path(self) -> Path:
        return self._output_dir_path() / "profile.json"

    def _chart_output_path(self) -> Path:
        return self._output_dir_path() / "profile_chart.png"

    def _render_plot(self) -> Optional[Path]:
        if not self.active:
            return None

        module_peaks = [
            module_stats
            for module_stats in self.stats["modules"].values()
            if module_stats["peak_delta_allocated_mib"] is not None
        ]
        snapshots = self._selected_snapshots()
        if not module_peaks and not snapshots and not self._timeline:
            return None

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            chart_path = self._chart_output_path()
            chart_path.parent.mkdir(parents=True, exist_ok=True)

            figure, axes = plt.subplots(4, 1, figsize=(14, 20), constrained_layout=True)

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

            if self._timeline:
                n = len(self._timeline)
                seqs = list(range(n))
                alloc_mib = [entry["allocated_mib"] for entry in self._timeline]
                res_mib = [entry["reserved_mib"] for entry in self._timeline]
                axes[1].plot(seqs, alloc_mib, linewidth=1, label="allocated")
                axes[1].plot(seqs, res_mib, linewidth=1, label="reserved")
                step = max(1, n // 30)
                tick_idxs = list(range(0, n, step))
                if (n - 1) not in tick_idxs:
                    tick_idxs.append(n - 1)
                tick_labels = [_shorten_timeline_label(self._timeline[i]["label"]) for i in tick_idxs]
                axes[1].set_xticks(tick_idxs)
                axes[1].set_xticklabels(tick_labels, rotation=75, ha="right", fontsize=7)
                axes[1].set_title("Per-Layer Memory Usage Timeline")
                axes[1].set_ylabel("MiB")
                axes[1].legend()
            else:
                axes[1].set_axis_off()

            if module_peaks:
                module_peaks.sort(key=lambda item: item["peak_delta_allocated_mib"], reverse=True)
                labels = [module_stats["name"] for module_stats in module_peaks]
                values = [module_stats["peak_delta_allocated_mib"] for module_stats in module_peaks]
                axes[2].bar(labels, values)
                axes[2].set_title("Module Peak Delta Allocated Memory")
                axes[2].set_ylabel("MiB")
                axes[2].tick_params(axis="x", rotation=75)
            else:
                axes[2].set_axis_off()

            module_times = [
                module_stats
                for module_stats in self.stats["modules"].values()
                if module_stats["avg_elapsed_ms"] is not None
            ]
            if module_times:
                module_times.sort(key=lambda item: item["avg_elapsed_ms"], reverse=True)
                labels = [module_stats["name"] for module_stats in module_times]
                values = [module_stats["avg_elapsed_ms"] for module_stats in module_times]
                axes[3].bar(labels, values)
                axes[3].set_title("Module Average Forward Time")
                axes[3].set_ylabel("ms")
                axes[3].tick_params(axis="x", rotation=75)
            else:
                axes[3].set_axis_off()

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
            chart_path = self.stats["artifacts"].get("chart")
            if chart_path:
                print(f"chart: {chart_path}", flush=True)
            elif self.stats["artifacts"].get("chart_error"):
                chart_error = self.stats["artifacts"]["chart_error"]
                print(f"chart error: {chart_error['type']}: {chart_error['message']}", flush=True)

    def finalize(self) -> Dict[str, Any]:
        ended_at = perf_counter()
        self.stats["timings"]["total_elapsed_ms"] = round((ended_at - self._started_at) * 1000.0, 4)

        if self._snapshot_marks:
            snapshot_durations: Dict[str, float] = {}
            for (prev_label, prev_t), (cur_label, cur_t) in zip(self._snapshot_marks[:-1], self._snapshot_marks[1:]):
                snapshot_durations[f"{prev_label}->{cur_label}"] = round((cur_t - prev_t) * 1000.0, 4)
            self.stats["timings"]["snapshot_durations_ms"] = snapshot_durations

        snapshots = self.stats.get("snapshots", {})
        before_forward = snapshots.get("before_model_forward", {}).get("elapsed_ms_from_start")
        after_forward = snapshots.get("after_model_forward", {}).get("elapsed_ms_from_start")
        if isinstance(before_forward, (float, int)) and isinstance(after_forward, (float, int)):
            self.stats["timings"]["model_forward_elapsed_ms"] = round(max(0.0, after_forward - before_forward), 4)

        module_total_elapsed_ms = 0.0
        slowest_module: Optional[Dict[str, Any]] = None
        for module_stats in self.stats["modules"].values():
            module_stats.pop("_call_start_perf_counter_s", None)
            elapsed = module_stats.get("total_elapsed_ms")
            if isinstance(elapsed, (float, int)):
                module_total_elapsed_ms += elapsed
            if module_stats.get("avg_elapsed_ms") is not None:
                if slowest_module is None or module_stats["avg_elapsed_ms"] > slowest_module["avg_elapsed_ms"]:
                    slowest_module = {
                        "name": module_stats["name"],
                        "avg_elapsed_ms": round(module_stats["avg_elapsed_ms"], 4),
                        "max_elapsed_ms": round(module_stats["max_elapsed_ms"], 4) if module_stats["max_elapsed_ms"] is not None else None,
                    }

        self.stats["timings"]["module_total_elapsed_ms"] = round(module_total_elapsed_ms, 4)
        self.stats["timings"]["slowest_module"] = slowest_module

        if self.active:
            overall_allocated_bytes = max(self._observed_allocated_values, default=0)
            overall_reserved_bytes = max(self._observed_reserved_values, default=0)
            self.stats["overall_peak"] = {
                "observed_peak_allocated_bytes": overall_allocated_bytes,
                "observed_peak_allocated_mib": _bytes_to_mib(overall_allocated_bytes),
                "observed_peak_reserved_bytes": overall_reserved_bytes,
                "observed_peak_reserved_mib": _bytes_to_mib(overall_reserved_bytes),
            }
            self.stats["timeline"] = self._timeline

            chart_path = self._render_plot()
            if chart_path is not None:
                self.stats["artifacts"]["chart"] = str(chart_path)

        if self.enabled:
            self._emit_report()

        if self.enabled:
            payload = json.dumps(self.stats, indent=2)
            output_file = self._json_output_path()
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(payload, encoding="utf-8")
            if self.report_level != "quiet":
                print(f"json: {output_file}", flush=True)

        return self.stats