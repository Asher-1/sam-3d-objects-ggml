#!/usr/bin/env python3
# Tiny-case conv3d parity harness: builds a minimal ss_decoder GGUF (4^3
# latent, 2 -> 3 channels, one res block, one upsample) with an INDEPENDENT
# gather/shuffle-table implementation, runs sam3d-cli, and compares the
# input_layer conv output against a numpy reference from the same GGUF.
import os
import struct
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from gguf import GGUFReader

sys.path.insert(0, "cpp_ggml/scripts")
from gguf_schema import add_common_kv, finish_gguf  # noqa: E402

import gguf  # noqa: E402

W, H, D, IC, OC = 4, 4, 4, 2, 3
rng = np.random.default_rng(7)


def pixel_shuffle_3d(x: torch.Tensor) -> torch.Tensor:
    """Mirror of sam3d pixel_shuffle_3d for (C*8, D, H, W) input."""
    C8, a, b, c = x.shape
    s = 2
    c_ = C8 // (s ** 3)
    y = x.reshape(c_, s, s, s, a, b, c)
    y = y.permute(0, 4, 1, 5, 2, 6, 3)  # (c_, a, s0, b, s1, c, s2)
    return y.reshape(c_, a * s, b * s, c * s)


def gidx(d, h, wd, kd, kh, kw):
    """x-coord taps with padding=1; out-of-range -> NP (zero sentinel row)."""
    rows = d * h * wd
    idx = np.empty(rows, dtype=np.int32)
    for d2 in range(d):
        for h2 in range(h):
            for w2 in range(wd):
                sd, sh, sw = d2 + kd - 1, h2 + kh - 1, w2 + kw - 1
                idx[(d2 * h + h2) * wd + w2] = (
                    (sd * h + sh) * wd + sw
                    if 0 <= sd < d and 0 <= sh < h and 0 <= sw < wd
                    else rows
                )
    return idx


def shuffle_table(c, d, h, wd):
    wo, ho, do = 2 * wd, 2 * h, 2 * d
    r2 = wo * ho * do
    r = d * h * wd
    idx = np.empty(c * r2, dtype=np.int32)
    for d2 in range(do):
        for h2 in range(ho):
            for w2 in range(wo):
                c8 = (d2 % 2) * 4 + (h2 % 2) * 2 + (w2 % 2)
                rconv = ((d2 // 2) * h + (h2 // 2)) * wd + (w2 // 2)
                base = (d2 * ho + h2) * wo + w2
                for ch in range(c):
                    idx[ch * r2 + base] = (ch * 8 + c8) * r + rconv
    return idx


out_path = "/tmp/tiny_conv.gguf"
w = gguf.GGUFWriter(out_path, "sam3d.ssdec")
add_common_kv(w, "ss_decoder", "f32")
w.add_uint32("dec.out_channels", 1)
w.add_uint32("dec.latent_channels", IC)
w.add_uint32("dec.num_res_blocks", 1)
w.add_uint32("dec.num_res_blocks_middle", 1)
w.add_array("dec.channels", [OC, OC])
w.add_string("dec.norm_type", "layer")

cw = {}  # remember weights for the numpy reference
def add_conv(name, oc, ic, scale=0.1):
    t = (rng.standard_normal((27, oc * ic)) * scale).astype(np.float32)
    cw[name] = t
    w.add_tensor(f"dec.{name}.weight", t)
    w.add_tensor(f"dec.{name}.bias", (rng.standard_normal(oc) * scale).astype(np.float32))

def add_res(name, ch, scale=0.1):
    w.add_tensor(f"dec.{name}.norm1.weight", (rng.standard_normal(ch) * 0.1 + 1).astype(np.float32))
    w.add_tensor(f"dec.{name}.norm1.bias", (rng.standard_normal(ch) * 0.1).astype(np.float32))
    add_conv(f"{name}.conv1", ch, ch, scale)
    w.add_tensor(f"dec.{name}.norm2.weight", (rng.standard_normal(ch) * 0.1 + 1).astype(np.float32))
    w.add_tensor(f"dec.{name}.norm2.bias", (rng.standard_normal(ch) * 0.1).astype(np.float32))
    add_conv(f"{name}.conv2", ch, ch, scale)

add_conv("input_layer", OC, IC, 1.0)
add_res("middle_block.0", OC)
add_res("blocks.0", OC)
add_conv("blocks.1.conv", OC * 8, OC)  # upsample: OC*8 channels on the 4^3 grid
add_res("blocks.2", OC)
w.add_tensor("dec.out_layer.0.weight", (rng.standard_normal(OC) * 0.1 + 1).astype(np.float32))
w.add_tensor("dec.out_layer.0.bias", (rng.standard_normal(OC) * 0.1).astype(np.float32))
add_conv("out_layer.2", 1, OC, 0.1)

g4 = lambda kd, kh, kw: gidx(D, H, W, kd, kh, kw)
g8 = lambda kd, kh, kw: gidx(2 * D, 2 * H, 2 * W, kd, kh, kw)
for kd in range(3):
    for kh in range(3):
        for kw in range(3):
            off = (kd * 3 + kh) * 3 + kw
            w.add_tensor(f"dec.gidx.input_layer.{off}", g4(kd, kh, kw))
            w.add_tensor(f"dec.gidx.middle_block_0.{off}", g4(kd, kh, kw))
            w.add_tensor(f"dec.gidx.blocks_0.{off}", g4(kd, kh, kw))
            w.add_tensor(f"dec.gidx.blocks_1.{off}", g4(kd, kh, kw))
            w.add_tensor(f"dec.gidx.blocks_2.{off}", g8(kd, kh, kw))
            w.add_tensor(f"dec.gidx.out_layer_2.{off}", g8(kd, kh, kw))
w.add_tensor("dec.shuffle.blocks_1", shuffle_table(OC, D, H, W))
finish_gguf(w)
print(f"wrote {out_path}")

# tiny latent
lat = rng.standard_normal((IC, D, H, W)).astype(np.float32)  # torch (C, D, H, W)
with open("/tmp/tiny_latent.samt", "wb") as f:
    f.write(struct.pack("<Ii", 0x544D4153, 4))
    f.write(struct.pack("<4q", W, H, D, IC))
    f.write(struct.pack("<i", 0))
    f.write(lat.tobytes())

for node in (0, 3, 5):  # 0: input_layer; 1: middle.0; 2: blocks.0; 3: upsample; 4: blocks.2; 5: final
    env = dict(os.environ, SAM3D_DEBUG_NODE=str(node), SAM3D_LOG_LEVEL="W")
    subprocess.run(["./cpp_ggml/build-cpu/bin/sam3d-cli", "decode-ss", "--model", out_path,
                    "--input", "/tmp/tiny_latent.samt", "--out", f"/tmp/tiny_node{node}.samt"],
                   check=True, env=env, capture_output=True)

    with open(f"/tmp/tiny_node{node}.samt", "rb") as f:
        data = f.read()
    magic, nd = struct.unpack("<Ii", data[:8])
    got_ne = struct.unpack(f"<{nd}q", data[8:8 + 8 * nd])
    got = np.frombuffer(data[12 + 8 * nd:], dtype="<f4").astype(np.float32)
    print(f"node {node} cpp got ne:", got_ne)

    rd = GGUFReader(out_path)
    tmap = {t.name: t for t in rd.tensors}

    def conv_ref(x, name, oc):
        w27 = tmap[f"dec.{name}.weight"].data.reshape(27, -1).astype(np.float32)
        ic = x.shape[0]
        bias = tmap[f"dec.{name}.bias"].data.astype(np.float32)
        wt = np.empty((oc, ic, 3, 3, 3), dtype=np.float32)
        for kd in range(3):
            for kh in range(3):
                for kw in range(3):
                    wt[:, :, kd, kh, kw] = w27[(kd * 3 + kh) * 3 + kw].reshape(oc, ic)
        return F.conv3d(x.unsqueeze(0), torch.from_numpy(wt),
                        torch.from_numpy(bias), padding=1)[0]

    def cln_ref(x, name):
        w = tmap[f"dec.{name}.weight"].data.astype(np.float32)
        b = tmap[f"dec.{name}.bias"].data.astype(np.float32)
        y = x.permute(1, 2, 3, 0)
        y = F.layer_norm(y, (y.shape[-1],), torch.from_numpy(w), torch.from_numpy(b), 1e-5)
        return y.permute(3, 0, 1, 2)

    def res_ref(x, name):
        h = cln_ref(x, f"{name}.norm1")
        h = F.silu(h)
        h = conv_ref(h, f"{name}.conv1", x.shape[0])
        h = cln_ref(h, f"{name}.norm2")
        h = F.silu(h)
        h = conv_ref(h, f"{name}.conv2", x.shape[0])
        return h + x

    x = torch.from_numpy(lat)
    h = conv_ref(x, "input_layer", OC)
    if node == 0:
        ref = h
    else:
        h = res_ref(h, "middle_block.0")
        h = res_ref(h, "blocks.0")
        # blocks.1: upsample conv (OC*8 on the 4^3 grid) + pixel shuffle
        w27 = tmap["dec.blocks.1.conv.weight"].data.reshape(27, OC * 8 * OC).astype(np.float32)
        bias = tmap["dec.blocks.1.conv.bias"].data.astype(np.float32)
        wt = np.empty((OC * 8, OC, 3, 3, 3), dtype=np.float32)
        for kd in range(3):
            for kh in range(3):
                for kw in range(3):
                    wt[:, :, kd, kh, kw] = w27[(kd * 3 + kh) * 3 + kw].reshape(OC * 8, OC)
        conv = F.conv3d(h.unsqueeze(0), torch.from_numpy(wt),
                        torch.from_numpy(bias), padding=1)[0]  # (OC*8, D, H, W)
        h = pixel_shuffle_3d(conv)
        if node == 3:
            ref = h
        else:
            h = res_ref(h, "blocks.2")
            h = cln_ref(h, "out_layer.0")
            h = F.silu(h)
            ref = conv_ref(h, "out_layer.2", 1)
    ref_flat = ref.reshape(-1).numpy() if hasattr(ref, "numpy") else ref.reshape(-1)

    d = np.abs(got - ref_flat)
    print(f"node {node}: max|d| = {d.max():.6f}  mean|d| = {d.mean():.6f}  "
          f"n_wrong(>1e-4) = {(d > 1e-4).sum()}/{d.size}")
    if d.max() > 1e-4:
        shape = ref.shape
        wrong = np.argwhere(d.reshape(shape) > 1e-4)
        print("wrong elements first 15:")
        for idx in wrong[:15]:
            print(f"  idx={tuple(idx)}  got={got.reshape(shape)[tuple(idx)]:.5f} "
                  f"ref={ref[tuple(idx)]:.5f}")
