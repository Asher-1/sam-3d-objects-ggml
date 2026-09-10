#!/usr/bin/env python3
"""Sync the local GGUF inventory with the Hugging Face weight repository.

The 2026-09-14 asset ruling (see docs/NATIVE_E2E_PARITY_CONTRACT.md
"Weight-Asset Rulings") deleted every Q4_0/Q4_1 GGUF and re-exported
`ss_generator-q8_0.gguf` (F16-pinned condition embedders) and
`slat_decoder_gs-{f16,q8_0,q4_k}.gguf` (mixed-fp16 contract). This script
mirrors that ruling to Asher-1/SAM_3D_OBJECTS_GGUF:

  deletes: ss_*/slat_*-q4_0.gguf and -q4_1.gguf (12 files)
  uploads: ss_generator-q8_0.gguf, slat_decoder_gs-{f16,q8_0,q4_k}.gguf

Usage:
  SAM3D_HF_TOKEN=hf_... python cpp_ggml/scripts/sync_hf_models.py --dry-run
  SAM3D_HF_TOKEN=hf_... python cpp_ggml/scripts/sync_hf_models.py

Without SAM3D_HF_TOKEN the script only prints the plan (read-only listing
works anonymously).
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

REPO_ID = "Asher-1/SAM_3D_OBJECTS_GGUF"
GGUF_DIR = Path(__file__).resolve().parents[1] / "models" / "gguf"

DELETE_PATTERNS = ("-q4_0.gguf", "-q4_1.gguf")
UPLOAD_FILES = (
    "ss_generator-q8_0.gguf",
    "slat_decoder_gs-f16.gguf",
    "slat_decoder_gs-q8_0.gguf",
    "slat_decoder_gs-q4_k.gguf",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan; never touch the remote repo")
    parser.add_argument("--repo", default=REPO_ID)
    args = parser.parse_args()

    from huggingface_hub import HfApi

    # HfApi resolves credentials on its own (HF_TOKEN env, then the stored
    # token file from `huggingface-cli login`); only require the env var as
    # an override, never as a gate.
    api = HfApi(token=os.environ.get("SAM3D_HF_TOKEN") or None)
    try:
        who = api.whoami()
        print(f"authenticated as: {who.get('name')}")
        authorized = True
    except Exception as error:  # noqa: BLE001 - auth is environment-specific
        print(f"not authenticated ({type(error).__name__}); read-only plan only")
        authorized = False
    remote = set(api.list_repo_files(args.repo, repo_type="model"))

    to_delete = sorted(name for name in remote
                       if name.endswith(DELETE_PATTERNS))
    to_upload = []
    for name in UPLOAD_FILES:
        local = GGUF_DIR / name
        if not local.is_file():
            raise SystemExit(f"error: missing local model {local}")
        if name in remote:
            # The remote hash needs a full download; force-reupload instead.
            to_upload.append((name, local, "reupload (remote hash not checked)"))
        else:
            to_upload.append((name, local, "new"))

    print(f"repo: {args.repo}")
    print(f"\ndelete ({len(to_delete)}):")
    for name in to_delete:
        print(f"  - {name}")
    print(f"\nupload ({len(to_upload)}):")
    for name, local, note in to_upload:
        print(f"  + {name}  ({local.stat().st_size / 1048576:.0f} MiB, {note})")

    if args.dry_run or not authorized:
        if not authorized and not args.dry_run:
            print("\nnot authenticated: dry run only.")
        return 0

    commit = "Asset ruling 2026-09-14: drop q4_0/q4_1, ship F16-pinned cemb ss_generator q8_0 and fixed-contract slat_decoder_gs"
    for name in to_delete:
        api.delete_file(name, args.repo, repo_type="model", commit_message=commit)
        print(f"deleted {name}")
    for name, local, _ in to_upload:
        api.upload_file(path_or_fileobj=str(local), path_in_repo=name,
                        repo_id=args.repo, repo_type="model",
                        commit_message=commit)
        print(f"uploaded {name}")
    print("sync complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
