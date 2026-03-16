import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from vggt.models.vggt import VGGT


VGGT_ENABLE_CUDA_MEM_STATS_ENV = "VGGT_ENABLE_CUDA_MEM_STATS"
VGGT_CUDA_MEM_STATS_FILE_ENV = "VGGT_CUDA_MEM_STATS_FILE"


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _bytes_to_mib(num_bytes: int) -> float:
    return round(num_bytes / (1024 ** 2), 4)


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
                "output_path": self.output_path,
            },
            "reason": None if self.active else self._inactive_reason(),
            "model": self._model_metadata(model),
            "inputs": {},
            "snapshots": {},
            "modules": {},
            "overall_peak": None,
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
            payload = json.dumps(self.stats, indent=2)
            if self.output_path:
                output_file = Path(self.output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_text(payload, encoding="utf-8")
            else:
                print(payload, flush=True)

        return self.stats