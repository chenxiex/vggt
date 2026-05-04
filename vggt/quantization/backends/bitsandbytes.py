from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from vggt.quantization.backends.base import QuantBackend
from vggt.quantization.config import QuantizationConfig, dtype_to_string
from vggt.quantization.smoothquant import (
    DEFAULT_ATTENTION_MODULE_PREFIXES,
    find_quantizable_linear_layers,
    load_smoothquant_artifact,
    normalize_scale_dict,
)


def _import_bitsandbytes() -> Any:
    try:
        return importlib.import_module("bitsandbytes")
    except ImportError as exc:
        raise ImportError(
            "bitsandbytes is required for the bitsandbytes quantization backend. "
            "Install it with `pip install 'vggt[bnb]'` or `pip install bitsandbytes`."
        ) from exc


def _artifact_meta(scales_or_artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = scales_or_artifact.get("meta")
    return meta if isinstance(meta, Mapping) else {}


def _copy_scaled_linear_weights(
    target: nn.Module,
    source: nn.Linear,
    smooth_scale: float,
) -> None:
    scale = float(smooth_scale)
    if scale <= 0.0:
        scale = 1.0

    with torch.no_grad():
        target_weight = getattr(target, "weight")
        scaled_weight = source.weight.detach().to(device=target_weight.device, dtype=target_weight.dtype) / scale
        target_weight.copy_(scaled_weight)

        target_bias = getattr(target, "bias", None)
        if source.bias is not None and target_bias is not None:
            target_bias.copy_(source.bias.detach().to(device=target_bias.device, dtype=target_bias.dtype))


def _is_under_replaced_layer(tensor_name: str, replaced_layers: set[str]) -> bool:
    return any(tensor_name == layer_name or tensor_name.startswith(f"{layer_name}.") for layer_name in replaced_layers)


def _cast_non_bnb_aggregator_tensors(model: Any, dtype: torch.dtype | None, replaced_layers: list[str]) -> int:
    if dtype is None:
        return 0

    aggregator = getattr(model, "aggregator", None)
    if aggregator is None:
        return 0

    excluded = set(replaced_layers)
    excluded.update(
        layer_name.removeprefix("aggregator.")
        for layer_name in replaced_layers
        if layer_name.startswith("aggregator.")
    )
    casted = 0

    with torch.no_grad():
        for name, parameter in aggregator.named_parameters(recurse=True):
            if _is_under_replaced_layer(name, excluded) or not parameter.is_floating_point():
                continue
            parameter.data = parameter.data.to(dtype=dtype)
            casted += 1

        for name, buffer in aggregator.named_buffers(recurse=True):
            if _is_under_replaced_layer(name, excluded) or not buffer.is_floating_point():
                continue
            buffer.data = buffer.data.to(dtype=dtype)
            casted += 1

    return casted


class BitsAndBytesSmoothQuantLinear(nn.Module):
    """bitsandbytes linear wrapper that applies SmoothQuant input scaling."""

    def __init__(
        self,
        bnb_linear: nn.Module,
        smooth_scale: float,
        weight_bits: int,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.bnb_linear = bnb_linear
        self.weight_bits = int(weight_bits)
        self.compute_dtype = compute_dtype
        self.in_features = int(getattr(bnb_linear, "in_features"))
        self.out_features = int(getattr(bnb_linear, "out_features"))
        self.register_buffer("smooth_scale", torch.tensor(float(smooth_scale), dtype=torch.float32))

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        smooth_scale: float = 1.0,
        quant_config: QuantizationConfig | Mapping[str, Any] | None = None,
    ) -> "BitsAndBytesSmoothQuantLinear":
        bnb = _import_bitsandbytes()
        config = QuantizationConfig.from_any(quant_config)
        weight_bits = int(config.weight_bits)
        bias = linear.bias is not None

        if weight_bits == 8:
            bnb_linear = bnb.nn.Linear8bitLt(
                linear.in_features,
                linear.out_features,
                bias=bias,
                has_fp16_weights=False,
            )
        elif weight_bits == 4:
            kwargs: dict[str, Any] = {
                "bias": bias,
                "compress_statistics": True,
                "quant_type": "nf4",
            }
            if config.compute_dtype is not None:
                kwargs["compute_dtype"] = config.compute_dtype
            bnb_linear = bnb.nn.Linear4bit(linear.in_features, linear.out_features, **kwargs)
        else:
            raise ValueError("bitsandbytes backend only supports weight_bits=4 or weight_bits=8")

        _copy_scaled_linear_weights(bnb_linear, linear, smooth_scale)
        return cls(
            bnb_linear=bnb_linear,
            smooth_scale=smooth_scale,
            weight_bits=weight_bits,
            compute_dtype=config.compute_dtype,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"weight_bits={self.weight_bits}, compute_dtype={dtype_to_string(self.compute_dtype)}, "
            f"smooth_scale={float(self.smooth_scale.item()):.6g}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smooth_scale = self.smooth_scale.to(device=x.device, dtype=x.dtype)
        return self.bnb_linear(x * smooth_scale)


class BitsAndBytesQuantBackend(QuantBackend):
    """bitsandbytes backend for frame/global attention and MLP linear layers."""

    def __init__(self) -> None:
        self._model = None
        self._summary: dict[str, Any] | None = None

    def prepare(self, model: Any, quant_config: Mapping[str, Any] | QuantizationConfig | None = None) -> Any:
        _import_bitsandbytes()
        config = QuantizationConfig.from_any(quant_config)

        scales: dict[str, float] = {}
        if config.smoothquant_path is not None:
            artifact = load_smoothquant_artifact(Path(config.smoothquant_path))
            config = config.with_artifact_meta(_artifact_meta(artifact))
            scales = normalize_scale_dict(artifact)

        if config.weight_bits not in {4, 8}:
            raise ValueError("bitsandbytes backend only supports weight_bits=4 or weight_bits=8")

        layers = find_quantizable_linear_layers(model, module_prefixes=DEFAULT_ATTENTION_MODULE_PREFIXES)
        missing: list[str] = []
        replaced: list[str] = []

        for layer_name, linear in layers.items():
            smooth_scale = scales.get(layer_name)
            if smooth_scale is None:
                missing.append(layer_name)
                if config.smoothquant_path is not None and config.smoothquant_strict:
                    continue
                smooth_scale = 1.0

            parent_name, child_name = layer_name.rsplit(".", 1) if "." in layer_name else ("", layer_name)
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(
                parent,
                child_name,
                BitsAndBytesSmoothQuantLinear.from_linear(
                    linear,
                    smooth_scale=smooth_scale,
                    quant_config=config,
                ),
            )
            replaced.append(layer_name)

        if config.smoothquant_path is not None and config.smoothquant_strict and missing:
            raise KeyError(
                "Missing SmoothQuant scales for quantizable frame/global linear layers: "
                + ", ".join(sorted(missing)[:10])
                + (" ..." if len(missing) > 10 else "")
            )

        casted_aggregator_tensors = _cast_non_bnb_aggregator_tensors(
            model,
            dtype=config.compute_dtype,
            replaced_layers=replaced,
        )
        unused = sorted(set(scales.keys()) - set(layers.keys()))
        self._summary = {
            "replaced": len(replaced),
            "missing": missing,
            "unused": unused,
            "casted_aggregator_tensors": casted_aggregator_tensors,
            "quant_config": config.to_meta(),
        }
        return model

    def convert(self, model_or_graph: Any) -> Any:
        self._model = model_or_graph
        return self._model

    def run(self, inputs: Any) -> Any:
        if self._model is None:
            raise RuntimeError("BitsAndBytesQuantBackend is not converted. Call convert() first.")
        return self._model(inputs)

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "name": "bitsandbytes",
            "bit_widths": ["w4", "w8"],
            "compute_dtypes": ["input", "float16", "bfloat16", "float32"],
            "operators": ["linear(qkv,proj,mlp.fc1,mlp.fc2)", "smoothquant_scale"],
            "devices": ["cuda"],
        }
