from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from vggt.quantization.backends.base import QuantBackend
from vggt.quantization.smoothquant import apply_smoothquant_w8a16, load_smoothquant_artifact


class PseudoQuantBackend(QuantBackend):
    """Pseudo quant backend based on existing SmoothQuant W8A16 replacement."""

    def __init__(self) -> None:
        self._model = None

    def prepare(self, model: Any, quant_config: Mapping[str, Any] | None = None) -> Any:
        quant_config = quant_config or {}
        smoothquant_path = quant_config.get("smoothquant_path")
        strict = bool(quant_config.get("smoothquant_strict", True))

        if smoothquant_path is not None:
            artifact = load_smoothquant_artifact(Path(smoothquant_path))
            apply_smoothquant_w8a16(model, artifact, strict=strict)

        return model

    def convert(self, model_or_graph: Any) -> Any:
        self._model = model_or_graph
        return self._model

    def run(self, inputs: Any) -> Any:
        if self._model is None:
            raise RuntimeError("PseudoQuantBackend is not converted. Call convert() first.")
        return self._model(inputs)

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "name": "pseudo",
            "bit_widths": ["w8a16"],
            "operators": ["linear(qkv,proj)"],
            "devices": ["cpu", "cuda"],
        }
