#!/usr/bin/env python3
"""Compare the production sparse-coordinate downsample to official CUDA PyTorch."""
from __future__ import annotations

import argparse
import ast
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from verify_pytorch_cuda_rng import reference_draws

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


SAMT_MAGIC = b"SAMT"
GGML_TYPE_I32 = 26


def official_downsample_function(torch):
    """Load the exact official function without importing unrelated pipeline modules."""
    source_path = REPOSITORY_ROOT / "sam3d_objects" / "pipeline" / "inference_utils.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(source_path))
    definition = next(
        (node for node in module.body
         if isinstance(node, ast.FunctionDef) and node.name == "downsample_sparse_structure"),
        None,
    )
    if definition is None:
        raise ValueError(f"{source_path}: missing downsample_sparse_structure")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["downsample_sparse_structure"]


def write_i32_samt(path: Path, values: bytes, columns: int) -> None:
    if len(values) != columns * 4 * 4:
        raise ValueError("coordinate payload must contain N x 4 int32 values")
    with path.open("wb") as stream:
        stream.write(SAMT_MAGIC)
        stream.write(struct.pack("<i", 2))
        stream.write(struct.pack("<2q", 4, columns))
        stream.write(struct.pack("<i", GGML_TYPE_I32))
        stream.write(values)


def read_i32_samt(path: Path) -> tuple[tuple[int, ...], bytes]:
    with path.open("rb") as stream:
        if stream.read(4) != SAMT_MAGIC:
            raise ValueError(f"{path}: invalid SAMT magic")
        (rank,) = struct.unpack("<i", stream.read(4))
        dimensions = struct.unpack(f"<{rank}q", stream.read(rank * 8))
        (ggml_type,) = struct.unpack("<i", stream.read(4))
        payload = stream.read()
    if rank != 2 or dimensions[0] != 4 or ggml_type not in (26, 30):
        raise ValueError(f"{path}: expected an I32 [4, N] SAMT tensor")
    expected_bytes = dimensions[0] * dimensions[1] * 4
    if len(payload) != expected_bytes:
        raise ValueError(f"{path}: malformed I32 payload")
    return dimensions, payload


def coordinate_fixture(torch):
    """Make a nondegenerate, unordered fixture that requires random selection."""
    rows: list[tuple[int, int, int, int]] = []
    for x in range(-17, 18):
        for y in range(-13, 14):
            for z in range(-11, 12):
                if (3 * x + 5 * y + 7 * z) % 4:
                    rows.append((0, x, y, z))
    # Exercise `torch.unique` sorting rather than relying on the source order.
    rows.extend(reversed(rows[:131]))
    rows.reverse()
    return torch.tensor(rows, dtype=torch.int32, device="cuda")


def find_distribution_blocks(binary: Path, seed: int, normal_draws: str, output: Path) -> int:
    subprocess.run(
        [str(binary), "rng-dump", "--implementation", "cuda", "--seed", str(seed),
         "--sizes", normal_draws, "--out-dir", str(output)],
        check=True,
    )
    value = json.loads((output / "rng_contract.json").read_text(encoding="utf-8")).get(
        "distribution_blocks"
    )
    if not isinstance(value, int) or value <= 0:
        raise ValueError("native CUDA RNG did not report a positive distribution_blocks value")
    return value


def native_downsample(binary: Path, source: Path, output: Path, seed: int, blocks: int,
                      normal_draws: str, max_coords: int, factor: int) -> bytes:
    subprocess.run(
        [str(binary), "coords-downsample", "--input", str(source), "--out", str(output),
         "--seed", str(seed), "--distribution-blocks", str(blocks),
         "--normal-draws", normal_draws, "--max-coords", str(max_coords),
         "--downsample-factor", str(factor)],
        check=True,
    )
    return read_i32_samt(output)


def first_row_difference(expected: bytes, actual: bytes) -> str:
    if len(expected) != len(actual):
        return f"payload size differs: expected {len(expected)}, actual {len(actual)}"
    for offset in range(0, len(expected), 16):
        if expected[offset:offset + 16] != actual[offset:offset + 16]:
            index = offset // 16
            expected_row = struct.unpack("<4i", expected[offset:offset + 16])
            actual_row = struct.unpack("<4i", actual[offset:offset + 16])
            return f"first differing row {index}: expected {expected_row}, actual {actual_row}"
    return "payloads differ without an addressable int32 row"


def row_set_difference(expected: bytes, actual: bytes) -> str:
    expected_rows = set(struct.iter_unpack("<4i", expected))
    actual_rows = set(struct.iter_unpack("<4i", actual))
    overlap = len(expected_rows & actual_rows)
    return (f"row-set overlap {overlap}/{len(expected_rows)} expected and "
            f"{len(actual_rows)} actual")


def row_bounds(payload: bytes) -> str:
    rows = list(struct.iter_unpack("<4i", payload))
    if not rows:
        return "empty"
    bounds = tuple((min(row[axis] for row in rows), max(row[axis] for row in rows))
                   for axis in range(1, 4))
    return f"rows={len(rows)}, xyz-bounds={bounds}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-binary", type=Path, required=True)
    parser.add_argument("--portable-binary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-coords", type=int, default=37)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument(
        "--normal-draws", default="6,3,32768,3,1",
        help="normal draw sizes preceding sparse-coordinate selection",
    )
    parser.add_argument(
        "--reference-dir", type=Path,
        help="fresh official stage oracle from which the preceding SS draws are derived",
    )
    args = parser.parse_args()
    for label, binary in (("CUDA", args.cuda_binary), ("portable", args.portable_binary)):
        if not binary.is_file():
            parser.error(f"missing {label} binary: {binary}")
    if args.max_coords <= 0 or args.downsample_factor <= 0:
        parser.error("--max-coords and --downsample-factor must be positive")

    try:
        import torch
    except ImportError as error:
        print(f"SKIP: PyTorch CUDA oracle is unavailable: {error}", file=sys.stderr)
        return 77
    if not torch.cuda.is_available():
        print("SKIP: CUDA PyTorch is unavailable", file=sys.stderr)
        return 77
    downsample_sparse_structure = official_downsample_function(torch)

    if args.reference_dir is not None:
        stage_draws, _ = reference_draws(args.reference_dir)
        if len(stage_draws) < 2:
            raise ValueError(f"{args.reference_dir}: expected SS and SLat random draws")
        normal_draws = stage_draws[:-1]
    else:
        normal_draws = [int(value) for value in args.normal_draws.split(",")]
    if not normal_draws or any(value <= 0 for value in normal_draws):
        parser.error("--normal-draws must be positive comma-separated integers")
    normal_draws_text = ",".join(str(value) for value in normal_draws)

    coordinates = coordinate_fixture(torch)
    if coordinates.shape[0] <= args.max_coords:
        raise AssertionError("fixture must exceed max-coords")
    candidate_limit = int(coordinates.shape[0]) - 1
    candidate_expected, candidate_factor = downsample_sparse_structure(
        coordinates, max_coords=candidate_limit, downsample_factor=args.downsample_factor
    )
    if candidate_factor != args.downsample_factor or candidate_expected.shape[0] >= candidate_limit:
        raise AssertionError("fixture must downsample coordinates without needing random selection")

    torch.manual_seed(args.seed)
    for count in normal_draws:
        torch.randn(count, device="cuda", dtype=torch.float32)
    expected, returned_factor = downsample_sparse_structure(
        coordinates, max_coords=args.max_coords, downsample_factor=args.downsample_factor
    )
    if returned_factor != args.downsample_factor or expected.shape != (args.max_coords, 4):
        raise AssertionError("official coordinate fixture did not exercise random downsampling")
    expected_payload = expected.cpu().contiguous().numpy().tobytes()
    candidate_expected_payload = candidate_expected.cpu().contiguous().numpy().tobytes()
    source_payload = coordinates.cpu().contiguous().numpy().tobytes()

    with tempfile.TemporaryDirectory(prefix="sam3d-pytorch-coordinate-downsample-") as directory:
        root = Path(directory)
        source = root / "coordinates.samt"
        write_i32_samt(source, source_payload, int(coordinates.shape[0]))
        blocks = find_distribution_blocks(
            args.cuda_binary, args.seed, normal_draws_text, root / "rng-contract"
        )
        candidate_dimensions, candidate_payload = native_downsample(
            args.cuda_binary, source, root / "candidates.samt", args.seed, blocks,
            normal_draws_text, candidate_limit, args.downsample_factor,
        )
        if candidate_dimensions != (4, int(candidate_expected.shape[0])):
            raise AssertionError(
                "native coordinate candidate shape differs from official PyTorch: "
                f"expected {(4, int(candidate_expected.shape[0]))} ({row_bounds(candidate_expected_payload)}), "
                f"actual {candidate_dimensions} ({row_bounds(candidate_payload)})"
            )
        if candidate_payload != candidate_expected_payload:
            raise AssertionError(
                "production coordinate transform differs before randperm: "
                + first_row_difference(candidate_expected_payload, candidate_payload) + "; "
                + row_set_difference(candidate_expected_payload, candidate_payload)
            )

        cuda_dimensions, cuda_payload = native_downsample(
            args.cuda_binary, source, root / "cuda.samt", args.seed, blocks,
            normal_draws_text, args.max_coords, args.downsample_factor,
        )
        portable_dimensions, portable_payload = native_downsample(
            args.portable_binary, source, root / "portable.samt", args.seed, blocks,
            normal_draws_text, args.max_coords, args.downsample_factor,
        )
    if cuda_dimensions != (4, args.max_coords) or portable_dimensions != cuda_dimensions:
        raise AssertionError("native coordinate random-selection shape differs from official PyTorch")
    if cuda_payload != expected_payload:
        raise AssertionError(
            "CUDA production coordinate downsample differs from official PyTorch: "
            + first_row_difference(expected_payload, cuda_payload) + "; "
            + row_set_difference(expected_payload, cuda_payload)
        )
    if portable_payload != expected_payload:
        raise AssertionError(
            "portable production coordinate downsample differs from official PyTorch: "
            + first_row_difference(expected_payload, portable_payload) + "; "
            + row_set_difference(expected_payload, portable_payload)
        )
    print(json.dumps({
        "distribution_blocks": blocks,
        "downsample_factor": args.downsample_factor,
        "max_coords": args.max_coords,
        "normal_draw_sizes": normal_draws,
        "passed": True,
        "rows_before": int(coordinates.shape[0]),
        "seed": args.seed,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
