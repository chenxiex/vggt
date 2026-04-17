import math
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from vggt.layers.attention import Attention

DEFAULT_WEIGHT_QMAX = 127.0
DEFAULT_ACT_QMAX = 65504.0
EPS = 1e-8


def find_attention_linear_layers(model: nn.Module) -> Dict[str, nn.Linear]:
    """Return all qkv/proj Linear layers inside attention modules."""
    layers: Dict[str, nn.Linear] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, Attention):
            continue
        for linear_name in ("qkv", "proj"):
            linear = getattr(module, linear_name, None)
            if isinstance(linear, nn.Linear):
                full_name = f"{module_name}.{linear_name}" if module_name else linear_name
                layers[full_name] = linear
    return layers


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().reshape(-1)[0].item())
    return float(value)


def normalize_scale_dict(scales_or_artifact: Mapping[str, Any]) -> Dict[str, float]:
    """Normalize a scale mapping or an artifact dict to {layer_name: float_scale}."""
    if "scales" in scales_or_artifact and isinstance(scales_or_artifact["scales"], Mapping):
        scales_obj: Mapping[str, Any] = scales_or_artifact["scales"]
    else:
        scales_obj = scales_or_artifact

    scales: Dict[str, float] = {}
    for name, value in scales_obj.items():
        scale = _to_float(value)
        if not math.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        scales[str(name)] = scale
    return scales


def load_smoothquant_artifact(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SmoothQuant artifact not found: {path}")

    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, Mapping):
        raise TypeError(f"SmoothQuant artifact should be a mapping, got {type(obj)}")

    artifact = dict(obj)
    if "scales" in artifact and isinstance(artifact["scales"], Mapping):
        artifact["scales"] = normalize_scale_dict(artifact["scales"])
    else:
        artifact = {"scales": normalize_scale_dict(artifact)}
    return artifact


def save_smoothquant_artifact(artifact: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(artifact), path)


def compute_smooth_scale(
    weight_max: float,
    act_max: float,
    weight_qmax: float = DEFAULT_WEIGHT_QMAX,
    act_qmax: float = DEFAULT_ACT_QMAX,
    eps: float = EPS,
) -> float:
    """
    Solve scale from:
        (weight_max / scale) / (act_max * scale) = weight_qmax / act_qmax
    """
    weight_max = max(float(weight_max), 0.0)
    act_max = max(float(act_max), 0.0)

    if weight_max <= eps or act_max <= eps:
        return 1.0

    ratio = float(weight_qmax) / float(act_qmax)
    denom = max(act_max * ratio, eps)
    scale = math.sqrt(weight_max / denom)
    if not math.isfinite(scale) or scale <= eps:
        return 1.0
    return float(scale)


def calibrate_attention_scales(
    model: nn.Module,
    run_calibration: Callable[[], None],
    weight_qmax: float = DEFAULT_WEIGHT_QMAX,
    act_qmax: float = DEFAULT_ACT_QMAX,
    eps: float = EPS,
) -> Dict[str, Any]:
    """
    Collect activation maxima by running `run_calibration` once and compute SmoothQuant scales.

    `run_calibration` should execute model forward(s) under eval/no_grad.
    """
    layers = find_attention_linear_layers(model)
    act_max: Dict[str, float] = {name: 0.0 for name in layers}
    handles = []

    for layer_name, linear in layers.items():

        def _pre_hook(module: nn.Module, inputs: tuple[Any, ...], name: str = layer_name):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            cur = float(x.detach().abs().amax().item())
            if cur > act_max[name]:
                act_max[name] = cur

        handles.append(linear.register_forward_pre_hook(_pre_hook))

    try:
        run_calibration()
    finally:
        for handle in handles:
            handle.remove()

    scales: Dict[str, float] = {}
    stats: Dict[str, Dict[str, float]] = {}

    for layer_name, linear in layers.items():
        weight_max = float(linear.weight.detach().abs().amax().item())
        layer_act_max = float(act_max[layer_name])
        scale = compute_smooth_scale(
            weight_max=weight_max,
            act_max=layer_act_max,
            weight_qmax=weight_qmax,
            act_qmax=act_qmax,
            eps=eps,
        )
        scales[layer_name] = scale
        stats[layer_name] = {
            "weight_max": weight_max,
            "act_max": layer_act_max,
            "scale": scale,
        }

    return {
        "meta": {
            "weight_qmax": float(weight_qmax),
            "act_qmax": float(act_qmax),
            "num_layers": len(layers),
        },
        "scales": scales,
        "stats": stats,
    }


class SmoothQuantW8A16Linear(nn.Module):
    """Linear layer with INT8 weight storage and SmoothQuant activation scaling."""

    def __init__(
        self,
        weight_int8: torch.Tensor,
        weight_scale: float,
        smooth_scale: float,
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        if weight_int8.dtype != torch.int8:
            raise TypeError(f"weight_int8 must be torch.int8, got {weight_int8.dtype}")

        self.in_features = int(weight_int8.shape[1])
        self.out_features = int(weight_int8.shape[0])

        self.register_buffer("weight_int8", weight_int8.contiguous())
        self.register_buffer("weight_scale", torch.tensor(float(weight_scale), dtype=torch.float32))
        self.register_buffer("smooth_scale", torch.tensor(float(smooth_scale), dtype=torch.float32))

        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().float().contiguous())

    @staticmethod
    def _quantize_weight(
        weight: torch.Tensor,
        smooth_scale: float,
        qmax: float,
        eps: float,
    ) -> tuple[torch.Tensor, float]:
        scaled_weight = weight.detach().float() / float(smooth_scale)
        max_abs = float(scaled_weight.abs().amax().item())
        if max_abs <= eps:
            weight_scale = 1.0
            weight_int8 = torch.zeros_like(scaled_weight, dtype=torch.int8)
        else:
            weight_scale = max(max_abs / float(qmax), eps)
            weight_int8 = torch.clamp(
                torch.round(scaled_weight / weight_scale),
                min=-int(qmax),
                max=int(qmax),
            ).to(torch.int8)
        return weight_int8, float(weight_scale)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        smooth_scale: float,
        qmax: float = DEFAULT_WEIGHT_QMAX,
        eps: float = EPS,
    ) -> "SmoothQuantW8A16Linear":
        weight_int8, weight_scale = cls._quantize_weight(
            weight=linear.weight,
            smooth_scale=smooth_scale,
            qmax=qmax,
            eps=eps,
        )
        return cls(
            weight_int8=weight_int8,
            weight_scale=weight_scale,
            smooth_scale=smooth_scale,
            bias=linear.bias,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"weight_dtype=int8, smooth_scale={float(self.smooth_scale.item()):.6g}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = x * self.smooth_scale.to(dtype=x.dtype, device=x.device)
        weight = self.weight_int8.to(dtype=x.dtype) * self.weight_scale.to(dtype=x.dtype, device=x.device)

        bias = None
        if self.bias is not None:
            bias = self.bias.to(dtype=x.dtype, device=x.device)

        return F.linear(x_scaled, weight, bias)


def apply_smoothquant_w8a16(
    model: nn.Module,
    scales_or_artifact: Mapping[str, Any],
    strict: bool = True,
    weight_qmax: float = DEFAULT_WEIGHT_QMAX,
    eps: float = EPS,
) -> Dict[str, Any]:
    """Replace attention qkv/proj linear layers with SmoothQuantW8A16Linear."""
    scales = normalize_scale_dict(scales_or_artifact)
    layers = find_attention_linear_layers(model)

    missing = []
    replaced = []

    for layer_name, linear in layers.items():
        if layer_name not in scales:
            missing.append(layer_name)
            continue

        parent_name, child_name = layer_name.rsplit(".", 1) if "." in layer_name else ("", layer_name)
        parent = model.get_submodule(parent_name) if parent_name else model
        quant_linear = SmoothQuantW8A16Linear.from_linear(
            linear,
            smooth_scale=scales[layer_name],
            qmax=weight_qmax,
            eps=eps,
        )
        setattr(parent, child_name, quant_linear)
        replaced.append(layer_name)

    if strict and missing:
        raise KeyError(
            "Missing SmoothQuant scales for attention linear layers: "
            + ", ".join(sorted(missing)[:10])
            + (" ..." if len(missing) > 10 else "")
        )

    unused = sorted(set(scales.keys()) - set(layers.keys()))
    return {
        "replaced": len(replaced),
        "missing": missing,
        "unused": unused,
    }
