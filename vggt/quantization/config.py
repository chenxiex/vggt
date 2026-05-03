from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import torch

DEFAULT_WEIGHT_BITS = 8
DEFAULT_ACTIVATION_BITS = 16
DEFAULT_ACTIVATION_QMAX_A16 = 65504.0

_PUBLIC_FIELDS = (
    "weight_bits",
    "activation_bits",
    "compute_dtype",
    "smoothquant_path",
    "smoothquant_strict",
    "weight_qmax",
    "activation_qmax",
)

_ALIASES = {
    "act_bits": "activation_bits",
    "activation_bit_width": "activation_bits",
    "weight_bit_width": "weight_bits",
    "act_qmax": "activation_qmax",
    "smoothquant_scale_path": "smoothquant_path",
}


def _parse_dtype(value: Any) -> torch.dtype | None:
    if value is None or value == "":
        return None
    if isinstance(value, torch.dtype):
        return value

    value_str = str(value).strip().lower()
    if value_str in {"none", "input", "auto"}:
        return None
    if value_str.startswith("torch."):
        value_str = value_str.removeprefix("torch.")

    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if value_str not in dtype_map:
        raise ValueError(
            "compute_dtype must be one of input, float32, float16, or bfloat16; "
            f"got {value!r}"
        )
    return dtype_map[value_str]


def dtype_to_string(dtype: torch.dtype | None) -> str:
    if dtype is None:
        return "input"
    if dtype is torch.float32:
        return "float32"
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.bfloat16:
        return "bfloat16"
    return str(dtype).removeprefix("torch.")


def _normalize_field_name(name: str) -> str:
    return _ALIASES.get(name, name)


@dataclass
class QuantizationConfig:
    weight_bits: int = DEFAULT_WEIGHT_BITS
    activation_bits: int = DEFAULT_ACTIVATION_BITS
    compute_dtype: torch.dtype | str | None = None
    smoothquant_path: str | Path | None = None
    smoothquant_strict: bool = True
    weight_qmax: float | None = None
    activation_qmax: float | None = None
    _explicit_fields: frozenset[str] = field(default_factory=frozenset, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.weight_bits = int(self.weight_bits)
        self.activation_bits = int(self.activation_bits)
        self.compute_dtype = _parse_dtype(self.compute_dtype)
        self.smoothquant_strict = bool(self.smoothquant_strict)

        if self.smoothquant_path is not None:
            self.smoothquant_path = Path(self.smoothquant_path)

        if self.weight_qmax is not None:
            self.weight_qmax = float(self.weight_qmax)
        if self.activation_qmax is not None:
            self.activation_qmax = float(self.activation_qmax)

        self._validate_bits()

        if not self._explicit_fields:
            self._explicit_fields = frozenset(_PUBLIC_FIELDS)

    @classmethod
    def from_any(cls, value: Any) -> "QuantizationConfig":
        if value is None:
            return cls._from_values({}, explicit_fields=set())

        if isinstance(value, QuantizationConfig):
            return value

        if isinstance(value, Mapping):
            return cls.from_mapping(value)

        data: dict[str, Any] = {}
        explicit_fields: set[str] = set()
        for field_name in _PUBLIC_FIELDS:
            if hasattr(value, field_name):
                data[field_name] = getattr(value, field_name)
                explicit_fields.add(field_name)
        if not data:
            raise TypeError(f"Unsupported quantization config type: {type(value)}")
        return cls._from_values(data, explicit_fields=explicit_fields)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "QuantizationConfig":
        data: dict[str, Any] = {}
        explicit_fields: set[str] = set()
        for key, val in mapping.items():
            field_name = _normalize_field_name(str(key))
            if field_name not in _PUBLIC_FIELDS:
                continue
            data[field_name] = val
            explicit_fields.add(field_name)
        return cls._from_values(data, explicit_fields=explicit_fields)

    @classmethod
    def _from_values(cls, data: Mapping[str, Any], explicit_fields: set[str]) -> "QuantizationConfig":
        cfg = cls(**dict(data))
        cfg._explicit_fields = frozenset(explicit_fields)
        return cfg

    def with_artifact_meta(self, meta: Mapping[str, Any] | None) -> "QuantizationConfig":
        if not meta:
            return self

        artifact_config = QuantizationConfig.from_mapping(meta)
        values = self.to_dict()

        for field_name in artifact_config._explicit_fields:
            if field_name in {"smoothquant_path", "smoothquant_strict"}:
                continue
            if field_name not in self._explicit_fields:
                values[field_name] = getattr(artifact_config, field_name)

        merged = QuantizationConfig._from_values(values, explicit_fields=set(self._explicit_fields))
        return merged

    def with_updates(self, **updates: Any) -> "QuantizationConfig":
        normalized_updates = {
            _normalize_field_name(key): value for key, value in updates.items() if value is not None
        }
        cfg = replace(self, **normalized_updates)
        cfg._explicit_fields = frozenset(set(self._explicit_fields) | set(normalized_updates))
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "compute_dtype": self.compute_dtype,
            "smoothquant_path": self.smoothquant_path,
            "smoothquant_strict": self.smoothquant_strict,
            "weight_qmax": self.weight_qmax,
            "activation_qmax": self.activation_qmax,
        }

    def to_meta(self) -> dict[str, Any]:
        weight_qmax = self.resolve_weight_qmax()
        activation_qmax = self.resolve_activation_qmax()
        return {
            "weight_bits": int(self.weight_bits),
            "activation_bits": int(self.activation_bits),
            "compute_dtype": dtype_to_string(self.compute_dtype),
            "weight_qmax": float(weight_qmax),
            "activation_qmax": float(activation_qmax),
            "act_qmax": float(activation_qmax),
            "activation_fake_quant": bool(self.activation_fake_quant),
        }

    @property
    def activation_fake_quant(self) -> bool:
        return self.activation_bits < 16

    def resolve_weight_qmax(self) -> float:
        if self.weight_qmax is not None:
            qmax = float(self.weight_qmax)
        else:
            qmax = float((1 << (self.weight_bits - 1)) - 1)
        if qmax <= 0.0 or qmax > 127.0:
            raise ValueError(f"weight_qmax must be in (0, 127], got {qmax}")
        return qmax

    def resolve_activation_qmax(self) -> float:
        if self.activation_qmax is not None:
            qmax = float(self.activation_qmax)
        elif self.activation_bits < 16:
            qmax = float((1 << (self.activation_bits - 1)) - 1)
        elif self.compute_dtype is not None and "compute_dtype" in self._explicit_fields:
            qmax = float(torch.finfo(self.compute_dtype).max)
        else:
            qmax = DEFAULT_ACTIVATION_QMAX_A16

        if qmax <= 0.0:
            raise ValueError(f"activation_qmax must be positive, got {qmax}")
        return qmax

    def _validate_bits(self) -> None:
        if self.weight_bits < 2 or self.weight_bits > 8:
            raise ValueError(f"weight_bits must be in [2, 8], got {self.weight_bits}")
        if self.activation_bits < 2 or self.activation_bits > 16:
            raise ValueError(
                f"activation_bits must be in [2, 16], got {self.activation_bits}"
            )
