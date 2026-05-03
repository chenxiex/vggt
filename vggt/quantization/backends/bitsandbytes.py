from __future__ import annotations

import logging
from typing import Any, Mapping

import torch
import torch.nn as nn

from vggt.quantization.backends.base import QuantBackend

logger = logging.getLogger(__name__)


class BitsandbytesQuantBackend(QuantBackend):
    """Bitsandbytes quant backend MVP using Linear8bitLt replacement."""

    def __init__(self) -> None:
        self._model: nn.Module | None = None
        self._converted = False
        self._target_keywords = ("qkv", "proj", "fc", "mlp")

    def prepare(self, model: Any, quant_config: Mapping[str, Any] | None = None) -> Any:
        self._model = model
        quant_config = quant_config or {}
        keywords = quant_config.get("target_linear_keywords")
        if keywords:
            self._target_keywords = tuple(str(k) for k in keywords)
        logger.info("[BitsandbytesQuantBackend] prepare done. target keywords=%s", self._target_keywords)
        return model

    def convert(self, model_or_graph: Any) -> Any:
        model = model_or_graph

        try:
            import bitsandbytes as bnb  # type: ignore
        except ImportError:
            logger.warning(
                "[BitsandbytesQuantBackend] bitsandbytes is not installed; fallback to FP16/FP32 for all layers."
            )
            self._model = model
            self._converted = True
            return model

        replaced, fallback = 0, 0
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(token in name.lower() for token in self._target_keywords):
                fallback += 1
                logger.info("[BitsandbytesQuantBackend] fallback layer=%s (not in target keywords)", name)
                continue

            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model

            qlinear = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                has_fp16_weights=True,
            )
            qlinear.weight.data.copy_(module.weight.data)
            if module.bias is not None and qlinear.bias is not None:
                qlinear.bias.data.copy_(module.bias.data)

            setattr(parent, child_name, qlinear)
            replaced += 1

        logger.info(
            "[BitsandbytesQuantBackend] convert done. replaced=%d, fallback_fp=%d",
            replaced,
            fallback,
        )

        self._model = model
        self._converted = True
        return model

    def run(self, inputs: Any) -> Any:
        if self._model is None or not self._converted:
            raise RuntimeError("BitsandbytesQuantBackend is not converted. Call convert() first.")
        return self._model(inputs)

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "name": "bitsandbytes",
            "bit_widths": ["int8"],
            "operators": ["linear(qkv,proj,fc,mlp)"],
            "devices": ["cuda"],
            "fallback": "unsupported layers remain FP16/FP32 with logging",
        }
