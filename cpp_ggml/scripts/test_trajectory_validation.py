#!/usr/bin/env python3
"""Unit tests for provenance and tensor-shape checks in trajectory reports."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from validate_slat_gguf_trajectory import (
    error as slat_error,
    verify_same_gguf_weight_scope as verify_slat_scope,
)
from validate_ss_gguf_trajectory import (
    compare_baseline,
    error as ss_error,
    verify_same_gguf_weight_scope as verify_ss_scope,
)


class TrajectoryValidationTest(unittest.TestCase):
    def test_ss_error_requires_exact_shape(self) -> None:
        reference = ((2, 2), np.zeros(4, dtype=np.float32))
        exact = ((2, 2), np.zeros(4, dtype=np.float32))
        self.assertEqual(ss_error(reference, exact)["max_abs"], 0.0)
        with self.assertRaisesRegex(ValueError, "tensor shape mismatch"):
            ss_error(reference, ((4,), np.zeros(4, dtype=np.float32)))

    def test_slat_error_requires_exact_shape(self) -> None:
        reference = ((8, 4), np.zeros(32, dtype=np.float32))
        exact = ((8, 4), np.zeros(32, dtype=np.float32))
        self.assertEqual(slat_error(reference, exact)["mae"], 0.0)
        with self.assertRaisesRegex(ValueError, "tensor shape mismatch"):
            slat_error(reference, ((4, 8), np.zeros(32, dtype=np.float32)))

    def test_same_gguf_receipt_must_name_candidate_model(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sam3d-trajectory-test-") as directory:
            root = Path(directory)
            model = root / "generator-q4_0.gguf"
            model.touch()
            receipt = {
                "schema": "sam3d.gguf_torch_backbone_loader.v1",
                "weight_scope": "same-gguf-dequantized",
                "gguf": str(model.resolve()),
                "backbone_tensors_loaded": 1,
            }
            (root / "reference_weight_scope.json").write_text(
                json.dumps(receipt), encoding="utf-8")
            self.assertEqual(
                verify_ss_scope(root, model),
                root / "reference_weight_scope.json")
            self.assertEqual(
                verify_slat_scope(root, model),
                root / "reference_weight_scope.json")
            receipt["gguf"] = str(root / "other.gguf")
            (root / "reference_weight_scope.json").write_text(
                json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different SS GGUF"):
                verify_ss_scope(root, model)

    def test_baseline_cannot_cross_weight_scopes(self) -> None:
        rows = [{
            "step": 1,
            "kind": "latent",
            "modality": "shape",
            "elements": 4,
            "mae": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "max_abs": 0.0,
        }]
        with tempfile.TemporaryDirectory(prefix="sam3d-trajectory-test-") as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(json.dumps({
                "reference_weight_scope": "official-checkpoint",
                "candidate_weight_scope": "gguf",
                "rows": rows,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reference weight scope"):
                compare_baseline(rows, path, "same-gguf-dequantized")


if __name__ == "__main__":
    unittest.main()
