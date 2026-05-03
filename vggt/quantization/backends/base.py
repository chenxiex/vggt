from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class QuantBackend(ABC):
    """Unified quantization backend interface."""

    @abstractmethod
    def prepare(self, model: Any, quant_config: Mapping[str, Any] | None = None) -> Any:
        """Prepare model/graph for quantization workflow."""

    @abstractmethod
    def convert(self, model_or_graph: Any) -> Any:
        """Convert prepared model/graph into deployable quantized representation."""

    @abstractmethod
    def run(self, inputs: Any) -> Any:
        """Execute inference with the converted backend artifact."""

    @abstractmethod
    def capabilities(self) -> Mapping[str, Any]:
        """Return backend capabilities: bit-widths, operators, and devices."""
