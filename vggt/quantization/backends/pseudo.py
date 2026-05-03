from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from vggt.quantization.backends.base import QuantBackend
from vggt.quantization.config import QuantizationConfig
from vggt.quantization.smoothquant import apply_smoothquant, load_smoothquant_artifact


class PseudoQuantBackend(QuantBackend):
    """Pseudo quant backend based on SmoothQuant linear replacement."""

    def __init__(self) -> None:
        self._model = None

    def prepare(self, model: Any, quant_config: Mapping[str, Any] | QuantizationConfig | None = None) -> Any:
        config = QuantizationConfig.from_any(quant_config)
        smoothquant_path = config.smoothquant_path

        if smoothquant_path is not None:
            artifact = load_smoothquant_artifact(Path(smoothquant_path))
            apply_smoothquant(model, artifact, quant_config=config)

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
            "bit_widths": ["w2-w8", "a2-a16"],
            "compute_dtypes": ["input", "float32", "float16", "bfloat16"],
            "operators": ["linear(qkv,proj)"],
            "devices": ["cpu", "cuda"],
        }
