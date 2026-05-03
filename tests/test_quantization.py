from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from vggt.layers.attention import Attention
from vggt.quantization import QuantizationConfig, SmoothQuantLinear, SmoothQuantW8A16Linear
from vggt.quantization.backends import PseudoQuantBackend
from vggt.quantization.smoothquant import load_smoothquant_artifact


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = Attention(dim=4, num_heads=2)


class _TinyAttentionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frame_blocks = nn.ModuleList([_TinyBlock()])


def _tiny_scales() -> dict[str, float]:
    return {
        "frame_blocks.0.attn.qkv": 1.0,
        "frame_blocks.0.attn.proj": 1.0,
    }


class QuantizationConfigTest(unittest.TestCase):
    def test_defaults_and_artifact_meta_merge(self) -> None:
        cfg = QuantizationConfig.from_any(None)
        self.assertEqual(cfg.weight_bits, 8)
        self.assertEqual(cfg.activation_bits, 16)
        self.assertEqual(cfg.resolve_weight_qmax(), 127.0)
        self.assertEqual(cfg.resolve_activation_qmax(), 65504.0)

        merged = cfg.with_artifact_meta(
            {"weight_bits": 4, "activation_bits": 8, "compute_dtype": "float32"}
        )
        self.assertEqual(merged.weight_bits, 4)
        self.assertEqual(merged.activation_bits, 8)
        self.assertEqual(merged.compute_dtype, torch.float32)
        self.assertEqual(merged.resolve_weight_qmax(), 7.0)
        self.assertEqual(merged.resolve_activation_qmax(), 127.0)

    def test_explicit_config_beats_artifact_meta(self) -> None:
        cfg = QuantizationConfig.from_mapping({"weight_bits": 8})
        merged = cfg.with_artifact_meta({"weight_bits": 4, "activation_bits": 8})
        self.assertEqual(merged.weight_bits, 8)
        self.assertEqual(merged.activation_bits, 8)


class SmoothQuantLinearTest(unittest.TestCase):
    def test_w8a16_matches_compat_wrapper(self) -> None:
        torch.manual_seed(0)
        linear = nn.Linear(4, 3)
        x = torch.randn(2, 5, 4)

        compat = SmoothQuantW8A16Linear.from_linear(linear, smooth_scale=1.25)
        generic = SmoothQuantLinear.from_linear(
            linear,
            smooth_scale=1.25,
            quant_config={"weight_bits": 8, "activation_bits": 16},
        )

        self.assertTrue(torch.allclose(compat(x), generic(x), atol=0.0, rtol=0.0))

    def test_w4a8_forward_shape_and_dtype(self) -> None:
        torch.manual_seed(1)
        linear = nn.Linear(4, 3)
        x = torch.randn(2, 5, 4)
        quant = SmoothQuantLinear.from_linear(
            linear,
            smooth_scale=0.75,
            quant_config={"weight_bits": 4, "activation_bits": 8, "compute_dtype": "float32"},
        )

        y = quant(x)
        self.assertEqual(y.shape, (2, 5, 3))
        self.assertEqual(y.dtype, torch.float32)
        self.assertEqual(int(quant.weight_int8.abs().amax().item()), 7)


class SmoothQuantArtifactAndBackendTest(unittest.TestCase):
    def test_load_old_and_new_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            old_path = Path(tmpdir) / "old.pt"
            new_path = Path(tmpdir) / "new.pt"
            torch.save({"layer": torch.tensor(2.0)}, old_path)
            torch.save(
                {
                    "meta": {"weight_bits": 4, "activation_bits": 8},
                    "scales": {"layer": torch.tensor(3.0)},
                },
                new_path,
            )

            old_artifact = load_smoothquant_artifact(old_path)
            new_artifact = load_smoothquant_artifact(new_path)

        self.assertEqual(old_artifact["scales"], {"layer": 2.0})
        self.assertEqual(new_artifact["meta"]["weight_bits"], 4)
        self.assertEqual(new_artifact["scales"], {"layer": 3.0})

    def test_backend_uses_artifact_meta_and_explicit_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "scales.pt"
            torch.save(
                {
                    "meta": {"weight_bits": 4, "activation_bits": 8},
                    "scales": _tiny_scales(),
                },
                artifact_path,
            )

            model = _TinyAttentionModel()
            backend = PseudoQuantBackend()
            backend.prepare(model, {"smoothquant_path": artifact_path})
            qkv = model.frame_blocks[0].attn.qkv
            self.assertIsInstance(qkv, SmoothQuantLinear)
            self.assertEqual(qkv.weight_bits, 4)
            self.assertEqual(qkv.activation_bits, 8)
            y = model.frame_blocks[0].attn(torch.randn(1, 3, 4))
            self.assertEqual(y.shape, (1, 3, 4))

            model = _TinyAttentionModel()
            backend.prepare(
                model,
                {
                    "smoothquant_path": artifact_path,
                    "weight_bits": 8,
                    "activation_bits": 16,
                },
            )
            qkv = model.frame_blocks[0].attn.qkv
            self.assertIsInstance(qkv, SmoothQuantLinear)
            self.assertEqual(qkv.weight_bits, 8)
            self.assertEqual(qkv.activation_bits, 16)


if __name__ == "__main__":
    unittest.main()
