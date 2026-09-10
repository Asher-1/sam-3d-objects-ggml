#!/usr/bin/env python3
"""CLI-level tests for the image-to-3d session batch mode.

These exercise argument validation and the batch manifest/error contract on
the fast failure paths (they never run a full reconstruction), plus the
manifest schema emitted by a completed batch when one is already present.
"""
from pathlib import Path
import json
import struct
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
VULKAN_CLI = REPO / "cpp_ggml/build-vulkan/bin/sam3d-cli"
IMAGE = REPO / "notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png"


def make_mask(directory: Path, name: str, width: int = 8, height: int = 8) -> Path:
    """Write a minimal valid PNG (RGBA, all opaque)."""
    import zlib

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload +
                struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    raw = b"".join(b"\x00" + b"\x40\x80\xc0\xff" * width for _ in range(height))
    png = (b"\x89PNG\r\n\x1a\n" +
           chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)) +
           chunk(b"IDAT", zlib.compress(raw)) +
           chunk(b"IEND", b""))
    path = directory / name
    path.write_bytes(png)
    return path


def invoke(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(VULKAN_CLI), "image-to-3d", *arguments],
                          capture_output=True, text=True, timeout=120)


class BatchArgumentTests(unittest.TestCase):
    def test_mask_list_rejects_mask_and_requires_image_and_out_dir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sam3d-batch-test-") as directory:
            room = Path(directory)
            mask = make_mask(room, "0.png")
            mask_list = room / "list.txt"
            mask_list.write_text(f"0\t{mask}\n", encoding="utf-8")
            base = ["--model", "cpp_ggml/models/gguf",
                    "--image", str(IMAGE), "--mask-list", str(mask_list)]
            # --mask and --mask-list are mutually exclusive.
            result = invoke(*base, "--mask", str(mask), "--out-dir", str(room))
            self.assertEqual(result.returncode, 1)
            self.assertIn("mutually exclusive", result.stderr)
            # A list without an image cannot define the shared RGB input.
            result = invoke("--model", "cpp_ggml/models/gguf",
                            "--mask-list", str(mask_list), "--out-dir", str(room))
            self.assertEqual(result.returncode, 1)
            self.assertIn("requires --image", result.stderr)
            # The batch form writes obj_<ID> trees; it refuses to guess a
            # root from --out.
            result = invoke(*base, "--out", str(room / "out.ply"))
            self.assertEqual(result.returncode, 1)

    def test_mask_list_rejects_empty_duplicate_and_missing_entries(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sam3d-batch-test-") as directory:
            room = Path(directory)
            mask = make_mask(room, "0.png")
            base = ["--model", "cpp_ggml/models/gguf",
                    "--image", str(IMAGE), "--out-dir", str(room)]
            empty = room / "empty.txt"
            empty.write_text("# only a comment\n\n", encoding="utf-8")
            result = invoke(*base, "--mask-list", str(empty))
            self.assertEqual(result.returncode, 1)
            self.assertIn("no entries", result.stderr)

            duplicate = room / "duplicate.txt"
            duplicate.write_text(f"3\t{mask}\n3\t{mask}\n", encoding="utf-8")
            result = invoke(*base, "--mask-list", str(duplicate))
            self.assertEqual(result.returncode, 1)
            self.assertIn("repeats object ID", result.stderr)

            missing = room / "missing.txt"
            missing.write_text("3\tdoes-not-exist.png\n", encoding="utf-8")
            result = invoke(*base, "--mask-list", str(missing))
            self.assertEqual(result.returncode, 1)
            self.assertIn("missing mask file", result.stderr)

            malformed = room / "malformed.txt"
            malformed.write_text("no-tab-here\n", encoding="utf-8")
            result = invoke(*base, "--mask-list", str(malformed))
            self.assertEqual(result.returncode, 1)
            self.assertIn("TAB", result.stderr)

            negative = room / "negative.txt"
            negative.write_text("-2\t{0}\n".format(mask), encoding="utf-8")
            result = invoke(*base, "--mask-list", str(negative))
            self.assertEqual(result.returncode, 1)
            self.assertIn("non-numeric", result.stderr)


class BatchManifestSchemaTests(unittest.TestCase):
    """Validate the manifest contract against a completed batch, when present.

    The full-scene run lives in output/alignment-vk27; CI machines without it
    skip this test, but a manifest that exists must satisfy the schema -
    including the resume-critical digest fields.
    """

    def setUp(self) -> None:
        self.manifest_path = REPO / "output/alignment-vk27/batch_manifest.json"
        if not self.manifest_path.is_file():
            self.skipTest("no completed batch manifest available")

    def test_manifest_schema_and_digest_fields(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["format"], "sam3d-batch-manifest/1")
        self.assertTrue(manifest["image"])
        self.assertIn("image_digest", manifest)
        self.assertIn("moge_model_digest", manifest)
        config = manifest["config"]
        for key in ("models_dir", "backend", "dtype", "seed", "threads"):
            self.assertIn(key, config)
        self.assertEqual(len(manifest["objects"]), 27,
                         "the full-scene batch must cover all 27 masks")
        for obj in manifest["objects"]:
            self.assertIn(obj["status"], ("ok", "failed"))
            if obj["status"] == "ok":
                self.assertTrue(obj["ply_digest"])
                self.assertTrue(obj["pose_digest"])
                self.assertIn("moge_pointmap_reused", obj)
                self.assertGreater(obj["elapsed_s"], 0.0)
            else:
                self.assertTrue(obj.get("error"))
        # The first object pays the MoGe forward; every later request with the
        # same RGB must have reused the cached point map.
        ok_objects = [obj for obj in manifest["objects"] if obj["status"] == "ok"]
        if len(ok_objects) > 1:
            self.assertFalse(ok_objects[0]["moge_pointmap_reused"])
            self.assertTrue(all(obj["moge_pointmap_reused"] for obj in ok_objects[1:]))


if __name__ == "__main__":
    unittest.main()
