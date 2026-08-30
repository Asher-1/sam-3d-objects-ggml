#!/usr/bin/env python3
"""Validate that a MoGe GGUF was exported from the requested official checkpoint."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from gguf import GGUFReader


def scalar_string(reader: GGUFReader, name: str) -> str | None:
    field = reader.fields.get(name)
    if field is None or len(field.parts) < 1:
        return None
    value = field.parts[-1]
    if hasattr(value, "tobytes"):
        return value.tobytes().decode("utf-8")
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"missing GGUF model: {args.model}")
    if not args.checkpoint.is_file():
        parser.error(f"missing official checkpoint: {args.checkpoint}")

    reader = GGUFReader(str(args.model))
    model_kind = scalar_string(reader, "sam3d.model")
    source = scalar_string(reader, "moge.source")
    recorded_hash = scalar_string(reader, "moge.source_sha256")
    with args.checkpoint.open("rb") as checkpoint_stream:
        actual_hash = hashlib.file_digest(checkpoint_stream, "sha256").hexdigest()
    expected_tensors = 421
    if model_kind != "moge_vitl":
        raise RuntimeError(f"wrong SAM 3D model type: {model_kind!r}")
    if source != "Ruicheng/moge-vitl:model.pt":
        raise RuntimeError(f"wrong MoGe source identity: {source!r}")
    if recorded_hash != actual_hash:
        raise RuntimeError("MoGe GGUF source hash does not match the supplied checkpoint")
    if len(reader.tensors) != expected_tensors:
        raise RuntimeError(f"expected {expected_tensors} MoGe tensors, got {len(reader.tensors)}")
    required = {"moge.backbone.patch_embed.proj.weight", "moge.head.projects.0.weight"}
    present = {tensor.name for tensor in reader.tensors}
    missing = sorted(required - present)
    if missing:
        raise RuntimeError("missing MoGe tensors: " + ", ".join(missing))
    print(f"PASS: {args.model} matches {args.checkpoint} ({len(reader.tensors)} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
