#!/usr/bin/env python3
"""Negative tests for release evidence validation, not model parity tests."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import subprocess
import unittest

from publish_full_e2e import REQUIRED_COVERAGE, SCHEMA, TIMER, VARIANTS, pose_errors, sha256, validate_full_report
from verify_native_runtime import dependency_errors


class FullE2EReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="sam3d-report-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        # The validator checks evidence integrity, not GLB parsing; that has a separate native test.
        asset = self.root / "fixture.bin"
        asset.write_bytes(b"synthetic report integrity fixture")
        manifest = {"path": asset.name, "bytes": asset.stat().st_size, "sha256": sha256(asset)}
        rows = []
        for name in ["pytorch"] + [f"{backend}-{dtype}" for backend, dtype in VARIANTS]:
            elapsed = 100000 if name == "pytorch" else (40000 if name.endswith("q4_0") else 50000)
            rows.append({"id": name, "latency_ms": elapsed, "latency_samples_ms": [elapsed] * 3,
                         "samples": 3, "timer_contract": TIMER, "gpu_exclusive": True,
                         "ss_attention": "strict",
                         "assets": {key: copy.deepcopy(manifest) for key in ("glb", "texture", "pose")},
                         "asset_validation": {"passed": True},
                         "neural_render_mae": 0.001, "glb_quality_gate_passed": True,
                         "glb_metrics": {"frame_count": 60, "resolution": 512, "rgb_mae_linear": 0.001,
                                         "mask_iou": 0.999, "normal_max_angle_deg": 0.01,
                                         "depth_mae_ndc": 0.001}})
        self.report = {"schema": SCHEMA, "candidate_sampling_mode": "native-seed", "rows": rows,
                       "full_acceptance_coverage": dict.fromkeys(REQUIRED_COVERAGE, True),
                       "glb_thresholds": {"max_rgb_mae_linear": 0.01, "min_mask_iou": 0.99,
                                          "max_normal_angle_deg": 1, "max_depth_mae_ndc": 0.01}}

    def result(self) -> dict:
        return validate_full_report(self.report, self.root)

    def test_complete_synthetic_contract(self) -> None:
        self.assertTrue(self.result()["passed"])

    def test_missing_budgets_cannot_pass(self) -> None:
        self.report["glb_thresholds"] = {}
        self.assertFalse(self.result()["passed"])

    def test_missing_coverage_cannot_pass(self) -> None:
        self.report["full_acceptance_coverage"] = {}
        self.assertFalse(self.result()["passed"])

    def test_duplicate_variant_cannot_hide_missing_variant(self) -> None:
        self.report["rows"][-1] = copy.deepcopy(self.report["rows"][-2])
        self.assertFalse(self.result()["passed"])

    def test_missing_texture(self) -> None:
        self.report["rows"][1]["assets"]["texture"]["path"] = "absent.png"
        self.assertFalse(self.result()["passed"])

    def test_modified_asset(self) -> None:
        (self.root / "fixture.bin").write_bytes(b"changed")
        self.assertFalse(self.result()["passed"])

    def test_escape_path(self) -> None:
        self.report["rows"][1]["assets"]["glb"]["path"] = "../outside.glb"
        self.assertFalse(self.result()["passed"])

    def test_nonfinite_latency(self) -> None:
        self.report["rows"][1]["latency_ms"] = float("nan")
        self.assertFalse(self.result()["passed"])

    def test_no_fabricated_sample_count(self) -> None:
        self.report["rows"][1]["latency_samples_ms"] = [50000]
        self.assertFalse(self.result()["passed"])

    def test_quality_flag_does_not_override_actual_metrics(self) -> None:
        self.report["rows"][1]["glb_metrics"]["rgb_mae_linear"] = 0.9
        self.assertFalse(self.result()["passed"])

    def test_slow_sample_cannot_hide_behind_a_fast_median(self) -> None:
        self.report["rows"][1]["latency_samples_ms"] = [50000, 50000, 90000]
        self.assertFalse(self.result()["passed"])

    def test_reported_latency_must_match_samples(self) -> None:
        self.report["rows"][1]["latency_ms"] = 10000
        self.assertFalse(self.result()["passed"])

    def test_negative_or_boolean_error_cannot_pass(self) -> None:
        for value in (-0.1, False):
            with self.subTest(value=value):
                self.report["rows"][1]["neural_render_mae"] = value
                self.assertFalse(self.result()["passed"])

    def test_same_timer_and_raw_input_are_required(self) -> None:
        for key, value in (("candidate_sampling_mode", "official-noise-replay-diagnostic"),
                           ("schema", "old-gaussian-only-report")):
            with self.subTest(key=key):
                report = copy.deepcopy(self.report)
                report[key] = value
                self.assertFalse(validate_full_report(report, self.root)["passed"])

    def test_shared_gpu_is_not_release_evidence(self) -> None:
        self.report["rows"][1]["gpu_exclusive"] = False
        self.assertFalse(self.result()["passed"])

    def test_missing_or_normal_attention_cannot_pass(self) -> None:
        for value in (None, "normal"):
            with self.subTest(value=value):
                self.report["rows"][1]["ss_attention"] = value
                self.assertFalse(self.result()["passed"])

    def test_pose_quaternion_sign_does_not_change_rotation(self) -> None:
        reference = {"rotation_wxyz": [1, 0, 0, 0], "translation": [0, 0, 0], "scale": [1, 1, 1]}
        native = {"rotation_wxyz": [-1, 0, 0, 0], "translation": [0, 0, 0], "scale": [1, 1, 1]}
        self.assertEqual(pose_errors(reference, native),
                         {"rotation_angle_deg": 0.0, "translation_l2": 0.0, "scale_max_abs": 0})


class RootLauncherTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[2]

    def run_launcher(self, runner: str, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(self.root / f"run_{runner}.sh"), *args],
                              cwd="/tmp", capture_output=True, text=True, check=False)

    def test_help_outside_checkout(self) -> None:
        for runner in ("python", "ggml"):
            with self.subTest(runner=runner):
                result = self.run_launcher(runner, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("2500-step", result.stdout)

    def test_invalid_numeric_and_path_arguments_fail_before_gpu_use(self) -> None:
        for arguments in (("--seed", "2147483648"), ("--seed", "999999999999999999999999"),
                          ("--threads", "0"), ("--threads", "7"), ("--backend", "unknown"),
                          ("--dtype", "invalid"), ("--image", "/definitely/missing.png"),
                          ("--out-dir",), ("--out-dir", ""), ("--image", "--mask")):
            with self.subTest(arguments=arguments):
                self.assertNotEqual(self.run_launcher("ggml", *arguments).returncode, 0)

    def test_nonempty_output_with_spaces_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sam3d launcher test ") as directory:
            sentinel = Path(directory) / "existing.glb"
            sentinel.write_bytes(b"do not overwrite")
            result = self.run_launcher("ggml", "--out-dir", directory, "--skip-build")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("new or empty", result.stderr)
            self.assertEqual(sentinel.read_bytes(), b"do not overwrite")


class NativeDependencyTests(unittest.TestCase):
    def test_cuda_runtime_and_blas_are_allowed(self) -> None:
        self.assertFalse(dependency_errors("libcudart.so => /usr/lib/libcudart.so\nlibcublas.so"))

    def test_forbidden_or_missing_library_fails(self) -> None:
        for entry in ("libcudnn.so.9", "libtorch_cuda.so", "libpython3.11.so", "libvtk.so => not found"):
            with self.subTest(entry=entry):
                self.assertTrue(dependency_errors(entry))


class ModelPreparationTests(unittest.TestCase):
    scripts = Path(__file__).resolve().parent

    def invoke(self, script: str, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(self.scripts / script), *arguments],
                              cwd="/tmp", capture_output=True, text=True, check=False)

    def test_help_and_invalid_arguments_without_network(self) -> None:
        for script in ("download_gguf.sh", "prepare_moge_gguf.sh"):
            with self.subTest(script=script):
                self.assertEqual(self.invoke(script, "--help").returncode, 0)
                self.assertNotEqual(self.invoke(script, "--dtype").returncode, 0)
                self.assertNotEqual(self.invoke(script, "--dtype", "invalid").returncode, 0)
        self.assertNotEqual(self.invoke("download_gguf.sh", "--model", "invalid").returncode, 0)

    def test_moge_never_overwrites_existing_baseline_implicitly(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sam3d model preparation ") as directory:
            model = Path(directory) / "moge_vitl-f16.gguf"
            for data, expected_code in ((b"GGUF existing fixture", 0), (b"invalid", 1)):
                with self.subTest(data=data):
                    model.write_bytes(data)
                    result = self.invoke("prepare_moge_gguf.sh", "--output", directory)
                    self.assertEqual(result.returncode, expected_code, result.stderr)
                    self.assertEqual(model.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
