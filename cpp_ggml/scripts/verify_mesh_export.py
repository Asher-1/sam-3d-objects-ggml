#!/usr/bin/env python3
"""Export official mesh-decoder tensors with C++ and validate the GLB contract."""
from __future__ import annotations

import argparse
import io
import json
import struct
import subprocess
from pathlib import Path


def samt_tensor(path: Path) -> tuple[tuple[int, ...], int, bytes]:
    with path.open("rb") as stream:
        if stream.read(4) != b"SAMT":
            raise ValueError(f"{path}: invalid SAMT magic")
        (rank,) = struct.unpack("<i", stream.read(4))
        if rank != 2:
            raise ValueError(f"{path}: expected rank-2 tensor, got {rank}")
        shape = struct.unpack("<2q", stream.read(16))
        (ggml_type,) = struct.unpack("<i", stream.read(4))
        return shape, ggml_type, stream.read()


def samt_f32(path: Path) -> tuple[tuple[int, ...], bytes]:
    shape, ggml_type, data = samt_tensor(path)
    if ggml_type != 0:
        raise ValueError(f"{path}: expected F32 tensor, got ggml type {ggml_type}")
    return shape, data


def glb_document(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"{path}: truncated GLB")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if (magic, version, length) != (0x46546C67, 2, len(data)):
        raise ValueError(f"{path}: invalid GLB header")
    json_length, json_type = struct.unpack_from("<II", data, 12)
    if json_type != 0x4E4F534A:
        raise ValueError(f"{path}: missing JSON chunk")
    document = json.loads(data[20:20 + json_length].decode("utf-8").rstrip(" "))
    bin_offset = 20 + json_length
    bin_length, bin_type = struct.unpack_from("<II", data, bin_offset)
    if bin_type != 0x004E4942 or bin_offset + 8 + bin_length != len(data):
        raise ValueError(f"{path}: invalid BIN chunk")
    return document, data[bin_offset + 8:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--vertices", type=Path, required=True)
    parser.add_argument("--faces", type=Path, required=True)
    parser.add_argument("--attrs", type=Path)
    parser.add_argument("--uv", type=Path)
    parser.add_argument("--texture", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if (args.attrs is None) == (args.uv is None):
        parser.error("provide exactly one of --attrs or --uv --texture")
    if (args.uv is None) != (args.texture is None):
        parser.error("--uv and --texture must be provided together")
    for label, path in (("binary", args.binary), ("vertices", args.vertices),
                        ("faces", args.faces)):
        if not path.is_file():
            parser.error(f"missing {label}: {path}")
    vertices_shape, vertices_data = samt_f32(args.vertices)
    faces_shape, faces_type, _ = samt_tensor(args.faces)
    if faces_type not in (0, 26):
        raise ValueError(f"{args.faces}: expected F32 or I32 faces, got ggml type {faces_type}")
    if vertices_shape[0] != 3 or faces_shape[0] != 3:
        raise RuntimeError(
            f"unexpected official mesh tensors: vertices={vertices_shape}, "
            f"faces={faces_shape}"
        )

    pbr = args.uv is not None
    if pbr:
        assert args.texture is not None
        if not args.uv.is_file() or not args.texture.is_file():
            parser.error("missing PBR UV or texture input")
        uv_shape, uv_data = samt_f32(args.uv)
        if uv_shape != (2, vertices_shape[1]):
            raise RuntimeError(f"unexpected official UV tensor shape: {uv_shape}")
    else:
        assert args.attrs is not None
        if not args.attrs.is_file():
            parser.error(f"missing attrs: {args.attrs}")
        attrs_shape, _ = samt_f32(args.attrs)
        if attrs_shape != (6, vertices_shape[1]):
            raise RuntimeError(f"unexpected official attribute tensor shape: {attrs_shape}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    command = [str(args.binary), "mesh-export", "--vertices", str(args.vertices),
               "--faces", str(args.faces)]
    if pbr:
        command.extend(("--uv", str(args.uv), "--texture", str(args.texture)))
    else:
        command.extend(("--attrs", str(args.attrs)))
    command.extend(("--out", str(args.out)))
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    doc, binary = glb_document(args.out)
    primitive = doc["meshes"][0]["primitives"][0]
    expected_attributes = {"POSITION", "TEXCOORD_0"} if pbr else {"POSITION", "COLOR_0"}
    if set(primitive["attributes"]) != expected_attributes:
        raise RuntimeError(f"unexpected GLB attributes: {primitive['attributes']}")
    accessors = doc["accessors"]
    vertex_count = accessors[primitive["attributes"]["POSITION"]]["count"]
    index_count = accessors[primitive["indices"]]["count"]
    if vertex_count != vertices_shape[1] or index_count != faces_shape[1] * 3:
        raise RuntimeError(
            f"GLB topology differs from decoder output: vertices={vertex_count}/{vertices_shape[1]}, "
            f"indices={index_count}/{faces_shape[1] * 3}"
        )
    material = doc["materials"][0]["pbrMetallicRoughness"]
    if material["roughnessFactor"] != 1.0:
        raise RuntimeError("GLB material must retain official roughness=1.0")
    if pbr:
        if material.get("baseColorTexture", {}).get("index") != 0:
            raise RuntimeError("baked mesh export must reference its base-color texture")
        texture_accessor = accessors[primitive["attributes"]["TEXCOORD_0"]]
        texture_view = doc["bufferViews"][texture_accessor["bufferView"]]
        output_uv = binary[texture_view.get("byteOffset", 0):
                           texture_view.get("byteOffset", 0) + texture_view["byteLength"]]
        if output_uv != uv_data:
            raise RuntimeError("GLB TEXCOORD_0 values differ from the xatlas UV input")
        image_view_index = doc["images"][0]["bufferView"]
        image_view = doc["bufferViews"][image_view_index]
        embedded_png = binary[image_view.get("byteOffset", 0):
                              image_view.get("byteOffset", 0) + image_view["byteLength"]]
        from PIL import Image

        with Image.open(args.texture) as source, Image.open(io.BytesIO(embedded_png)) as embedded:
            if source.convert("RGBA").tobytes() != embedded.convert("RGBA").tobytes():
                raise RuntimeError("GLB embedded base-color PNG differs from native bake input")
    elif "baseColorTexture" in material:
        raise RuntimeError("non-baked mesh export must retain vertex colors without claiming a texture bake")
    first_input_vertex = struct.unpack_from("<3f", vertices_data, 0)
    position_view = doc["bufferViews"][accessors[primitive["attributes"]["POSITION"]]["bufferView"]]
    first_output_vertex = struct.unpack_from("<3f", binary, position_view.get("byteOffset", 0))
    expected_vertex = (first_input_vertex[0], first_input_vertex[2], -first_input_vertex[1])
    if any(abs(actual - expected) > 1.0e-6
           for actual, expected in zip(first_output_vertex, expected_vertex)):
        raise RuntimeError(
            f"GLB coordinate convention differs from official to_glb: "
            f"got={first_output_vertex}, expected={expected_vertex}"
        )
    kind = "baked PBR" if pbr else "vertex-color"
    print(f"PASS: native {kind} GLB preserves {vertex_count} vertices and {index_count // 3} triangles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
