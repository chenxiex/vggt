from __future__ import annotations

from vggt.quantization.backends.base import QuantBackend
from vggt.quantization.backends.bitsandbytes import BitsAndBytesQuantBackend
from vggt.quantization.backends.pseudo import PseudoQuantBackend


def create_quant_backend(name: str) -> QuantBackend:
    backend = name.lower()
    if backend == "pseudo":
        return PseudoQuantBackend()
    if backend == "bitsandbytes":
        return BitsAndBytesQuantBackend()

    if backend in {"torchao", "trt_int8"}:
        raise NotImplementedError(f"Quant backend '{backend}' is not implemented yet.")

    raise ValueError(f"Unknown quant backend: {name}")
