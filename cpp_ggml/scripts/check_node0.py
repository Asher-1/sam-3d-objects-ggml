#!/usr/bin/env python3
# Recompute node 0 (input_layer conv3d) from the GGUF weights + latent and
# compare against both the torch reference and the C++ dump. Locates whether
# the mismatch comes from the GGUF data or the C++ graph.
import struct
import numpy as np
import torch
import torch.nn.functional as F
from gguf import GGUFReader


def load_samt_flat(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, nd = struct.unpack("<Ii", data[:8])
    ne = struct.unpack(f"<{nd}q", data[8:8 + 8 * nd])
    (typ,) = struct.unpack("<i", data[8 + 8 * nd:12 + 8 * nd])
    payload = data[12 + 8 * nd:]
    arr = np.frombuffer(payload, dtype="<f4" if typ == 0 else "<f2")
    return ne, arr.astype(np.float32)


rd = GGUFReader("cpp_ggml/models/gguf/ss_decoder-f32.gguf")
tmap = {t.name: t for t in rd.tensors}
tw = tmap["dec.input_layer.weight"]  # ne {OC*IC, 27}
tb = tmap["dec.input_layer.bias"]
w27 = tw.data.astype(np.float32).reshape(27, 512 * 8) if tw.data.dtype != np.float32 else tw.data.reshape(27, 512 * 8)
print("weight ggml ne:", tw.data.shape, tw.data.dtype)
bias = tb.data.astype(np.float32)

# rebuild torch weight: wt[ch, i, kd, kh, kw] = w27[kd*9+kh*3+kw, ch*8+i]
wt = np.empty((512, 8, 3, 3, 3), dtype=np.float32)
for kd in range(3):
    for kh in range(3):
        for kw in range(3):
            off = (kd * 3 + kh) * 3 + kw
            wt[:, :, kd, kh, kw] = w27[off].reshape(512, 8)

ne, lat = load_samt_flat("/tmp/ss_latent.bin")
lat = lat.reshape(8, 16, 16, 16)  # torch (C, D, H, W)

with torch.no_grad():
    ref = F.conv3d(torch.from_numpy(lat).unsqueeze(0),
                   torch.from_numpy(wt), torch.from_numpy(bias), padding=1)
ref = ref[0].numpy().reshape(-1)

_, cpp = load_samt_flat("/tmp/ssgot/node_0.samt")
_, tor = load_samt_flat("/tmp/ssref/node_0.samt")

print(f"gguf-recomputed vs torch-ref: max|d| = {np.abs(ref - tor).max():.6f}")
print(f"cpp        vs torch-ref    : max|d| = {np.abs(cpp - tor).max():.6f}")
print(f"cpp        vs gguf-recomp  : max|d| = {np.abs(cpp - ref).max():.6f}")

# if cpp != gguf-recomp, check the spatial transpose hypothesis: maybe the
# cpp output is a (D,H,W) permutation of the reference
lr = ref.reshape(512, 16, 16, 16)
lc = cpp.reshape(512, 16, 16, 16)
for name, perm in [("dhw->hwd", (0, 2, 3, 1)), ("dhw->wdh", (0, 3, 1, 2)), ("dhw->wdh2", (0, 3, 2, 1)), ("dhw->hdw", (0, 2, 1, 3)), ("dhw->dwh", (0, 1, 3, 2))]:
    t = lc.permute(*[p - 1 if p else 0 for p in perm]) if False else lc
    pass
# channel-major vs position-major check
lc2 = cpp.reshape(16, 16, 16, 512).transpose(3, 0, 1, 2).reshape(-1)
print(f"cpp(as pos-major) vs gguf-recomp: max|d| = {np.abs(lc2 - ref).max():.6f}")
