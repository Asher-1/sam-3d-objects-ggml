#!/usr/bin/env python3
# Verify the C++ debug tensor dumps (xrows_p / g0 / part0) for the tiny conv
# test against numpy expectations computed from the tiny GGUF + latent.
import struct
import sys

import numpy as np
from gguf import GGUFReader

W = H = D = 4
IC, OC = 2, 3
NP = W * H * D


def load_samt_flat(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, nd = struct.unpack("<Ii", data[:8])
    ne = struct.unpack(f"<{nd}q", data[8:8 + 8 * nd])
    payload = data[12 + 8 * nd:]
    return ne, np.frombuffer(payload, dtype="<f4").astype(np.float32)


rd = GGUFReader("/tmp/tiny_conv.gguf")
tmap = {t.name: t for t in rd.tensors}

with open("/tmp/tiny_latent.samt", "rb") as f:
    data = f.read()
magic, nd = struct.unpack("<Ii", data[:8])
ne = struct.unpack(f"<{nd}q", data[8:8 + 8 * nd])
lat = np.frombuffer(data[12 + 8 * nd:], dtype="<f4").astype(np.float32)
print("latent ne:", ne)

name = sys.argv[1] if len(sys.argv) > 1 else "dbg.input_layer.xrows_p"
got_ne, got = load_samt_flat("/tmp/dbg_tensor.samt")
print(f"dbg tensor '{name}' ne: {got_ne} n={got.size}")

if name.endswith("xrows_p"):
    exp = np.concatenate([lat.reshape(IC, NP), np.zeros((IC, 1))], axis=1).reshape(-1)
    print("expect ne: (2, 65) n=", exp.size)
elif name.endswith("g0"):
    t = tmap["dec.gidx.input_layer.0"].data.astype(np.int32)
    xrows_p = np.concatenate([lat.reshape(IC, NP), np.zeros((IC, 1))], axis=1)
    exp = xrows_p[:, t].reshape(-1)
    print("expect ne: (2, 64) n=", exp.size)
elif name.endswith("part0"):
    t = tmap["dec.gidx.input_layer.0"].data.astype(np.int32)
    xrows_p = np.concatenate([lat.reshape(IC, NP), np.zeros((IC, 1))], axis=1)
    g = xrows_p[:, t]  # (IC, 64)
    w27 = tmap["dec.input_layer.weight"].data.reshape(27, OC * IC).astype(np.float32)
    wk = w27[0].reshape(OC, IC).T  # (IC, OC): wk[c, ch] = w27[0, ch*IC+c]
    exp = (wk @ g).T.reshape(-1)  # (64, OC) row-major
    print("expect ne: (64, 3) n=", exp.size)
else:
    sys.exit("unknown dbg tensor")

d = np.abs(got - exp)
print(f"max|d| = {d.max():.6f}  n_wrong(>1e-5) = {(d > 1e-5).sum()}/{d.size}")
if d.max() > 1e-5 and got.size == exp.size:
    bad = np.nonzero(d > 1e-5)[0][:15]
    for i in bad:
        print(f"  flat[{i}] got={got[i]:.5f} exp={exp[i]:.5f}")
