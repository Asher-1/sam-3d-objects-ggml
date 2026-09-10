#!/usr/bin/env python3
"""Exercise patch delivery in a disposable clone without touching the real index."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class PatchDeliveryTests(unittest.TestCase):
    def test_clean_replay_readonly_check_and_unknown_change_rejection(self) -> None:
        cpp = Path(__file__).resolve().parents[1]
        upstream = cpp / "third_party/ggml"
        revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        with tempfile.TemporaryDirectory(prefix="sam3d-patch-delivery-") as directory:
            fixture = Path(directory)
            scripts = fixture / "scripts"
            scripts.mkdir()
            helper = scripts / "apply_ggml_patches.sh"
            shutil.copy2(cpp / "scripts/apply_ggml_patches.sh", helper)
            patch_dir = fixture / "third_party/ggml-patches"
            patch_dir.mkdir(parents=True)
            shutil.copy2(cpp / "third_party/ggml-patches/0001-sam3d-ggml-combined.patch", patch_dir)
            clone = fixture / "third_party/ggml"
            subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(upstream), str(clone)], check=True)
            # --shared --no-checkout clones carry index entries that make a
            # plain checkout silently skip writing files; -f materialises the
            # tree the replay below needs.
            subprocess.run(["git", "-C", str(clone), "checkout", "--quiet", "--detach", "-f", revision], check=True)

            def invoke(*arguments: str) -> subprocess.CompletedProcess:
                return subprocess.run(["bash", str(helper), *arguments], capture_output=True, text=True, check=False)

            result = invoke("--check")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("source tree unchanged", result.stdout)
            clean = subprocess.check_output(["git", "-C", str(clone), "status", "--porcelain"], text=True)
            self.assertFalse(clean)
            result = invoke()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("applied 1", result.stdout)
            for arguments in ((), ("--check",)):
                result = invoke(*arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("final tree verified", result.stdout)
            extra = clone / "src/unrecorded-regression.cpp"
            extra.write_text("// intentionally absent from the delivered patch\n", encoding="utf-8")
            for arguments in ((), ("--check",)):
                result = invoke(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("not represented", result.stderr)
                self.assertTrue(extra.is_file())
            self.assertNotEqual(invoke("--unknown").returncode, 0)
            (patch_dir / "0002-unapplied.patch").write_text("unapplied extra patch\n", encoding="utf-8")
            result = invoke("--check")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exactly one", result.stderr)


if __name__ == "__main__":
    unittest.main()
