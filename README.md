# SAM 3D

SAM 3D Objects is one part of SAM 3D, a pair of models for object and human mesh reconstruction.  If you’re looking for SAM 3D Body, [click here](https://github.com/facebookresearch/sam-3d-body).

# SAM 3D Objects

**SAM 3D Team**, [Xingyu Chen](https://scholar.google.com/citations?user=gjSHr6YAAAAJ&hl=en&oi=sra)\*, [Fu-Jen Chu](https://fujenchu.github.io/)\*, [Pierre Gleize](https://scholar.google.com/citations?user=4imOcw4AAAAJ&hl=en&oi=ao)\*, [Kevin J Liang](https://kevinjliang.github.io/)\*, [Alexander Sax](https://alexsax.github.io/)\*, [Hao Tang](https://scholar.google.com/citations?user=XY6Nh9YAAAAJ&hl=en&oi=sra)\*, [Weiyao Wang](https://sites.google.com/view/weiyaowang/home)\*, [Michelle Guo](https://scholar.google.com/citations?user=lyjjpNMAAAAJ&hl=en&oi=ao), [Thibaut Hardin](https://github.com/Thibaut-H), [Xiang Li](https://ryanxli.github.io/)⚬, [Aohan Lin](https://github.com/linaohan), [Jia-Wei Liu](https://jia-wei-liu.github.io/), [Ziqi Ma](https://ziqi-ma.github.io/)⚬, [Anushka Sagar](https://www.linkedin.com/in/anushkasagar/), [Bowen Song](https://scholar.google.com/citations?user=QQKVkfcAAAAJ&hl=en&oi=sra)⚬, [Xiaodong Wang](https://scholar.google.com/citations?authuser=2&user=rMpcFYgAAAAJ), [Jianing Yang](https://jedyang.com/)⚬, [Bowen Zhang](http://home.ustc.edu.cn/~zhangbowen/)⚬, [Piotr Dollár](https://pdollar.github.io/)†, [Georgia Gkioxari](https://georgiagkioxari.com/)†, [Matt Feiszli](https://scholar.google.com/citations?user=A-wA73gAAAAJ&hl=en&oi=ao)†§, [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/)†§

***Meta Superintelligence Labs***

*Core contributor (Alphabetical, Equal Contribution), ⚬Intern, †Project leads, §Equal Contribution

[[`Paper`](https://ai.meta.com/research/publications/sam-3d-3dfy-anything-in-images/)] [[`Code`](https://github.com/facebookresearch/sam-3d-objects)] [[`Website`](https://ai.meta.com/sam3d/)] [[`Demo`](https://www.aidemos.meta.com/segment-anything/editor/convert-image-to-3d)] [[`Blog`](https://ai.meta.com/blog/sam-3d/)] [[`BibTeX`](#citing-sam-3d-objects)] [[`Roboflow`](https://blog.roboflow.com/sam-3d/)]

**SAM 3D Objects** is a foundation model that reconstructs full 3D shape geometry, texture, and layout from a single image, excelling in real-world scenarios with occlusion and clutter by using progressive training and a data engine with human feedback. It outperforms prior 3D generation models in human preference tests on real-world objects and scenes. We released code, weights, online demo, and a new challenging benchmark.


<p align="center"><img src="doc/intro.png"/></p>

-----

<p align="center"><img src="doc/arch.png"/></p>

## Latest updates

* **06/02/2026** - [3D Artist Object Set](https://ai.meta.com/datasets/sa-3dao-sam-3d-artist-objects/) and [HF Leaderboard](https://huggingface.co/spaces/facebook/sa3dao-leaderboard) are out.
* **06/01/2026** - Encoder weights are out.
* **11/19/2025** - Checkpoints Launched, Web Demo and Paper are out.

## Installation

Follow the [setup](doc/setup.md) steps before running the following.

### One-command textured GLB

The repository provides two root launchers for complete image/mask-to-textured
GLB reconstruction, including mesh cleanup, UVs, 100-view Gaussian observations,
2500-step texture baking and a standalone base-color PNG:

```bash
# Official Python; installs its environment when --setup is supplied.
bash run_python.sh --setup --out-dir output/python

# Native C++/GGML, including CUDA post-processing and embedded texture.
bash run_ggml.sh --backend cuda --dtype q8_0 \
  --accept-pbr-licenses --out-dir output/cuda-q8

# Vulkan neural inference with the same CUDA post-processing.
bash run_ggml.sh --backend vulkan --dtype q8_0 \
  --accept-pbr-licenses --out-dir output/vulkan-q8
```

See the [clone-to-run guide](cpp_ggml/README.md#clone-and-bootstrap) for CUDA,
VTK, OpenCV, model downloads and license prerequisites. Pass `--image`, `--mask`
and a new `--out-dir` for another object; use `--skip-build` for later native
runs. Models are not automatically downloaded without the required access.
Native output is `output.glb` plus `output.base_color.png`; Python output is
`official_pbr_00.glb` plus `official_pbr_00.base_color.png`. PBR here matches the
official baked base-color material, not independently predicted metallic,
roughness or normal maps. Actual parity status is reported below, not implied
by a successful export.

## Single or Multi-Object 3D Generation

SAM 3D Objects can convert masked objects in an image, into 3D models with pose, shape, texture, and layout. SAM 3D is designed to be robust in challenging natural images, handling small objects and occlusions, unusual poses, and difficult situations encountered in uncurated natural scenes like this kidsroom:

<p align="center">
  <img src="notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png" width="55%"/>
  <img src="doc/kidsroom_transparent.gif" width="40%"/>
</p>

For a quick start, run `python demo.py` or use the the following lines of code:

```python
import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image and mask
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# run model
output = inference(image, mask, seed=42)

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
```

For  more details and multi-object reconstruction, please take a look at out two jupyter notebooks:
* [single object](notebook/demo_single_object.ipynb)
* [multi object](notebook/demo_multi_object.ipynb)


## SAM 3D Body

[SAM 3D Body (3DB)](https://github.com/facebookresearch/sam-3d-body) is a robust promptable foundation model for single-image 3D human mesh recovery (HMR).

As a way to combine the strengths of both **SAM 3D Objects** and **SAM 3D Body**, we provide an example notebook that demonstrates how to combine the results of both models such that they are aligned in the same frame of reference. Check it out [here](notebook/demo_3db_mesh_alignment.ipynb).

## C++ (ggml) Runtime: performance and advantages

[`cpp_ggml/`](cpp_ggml/) provides a native GGML implementation of the
condition, SS, SLat and Gaussian stages. It supports CPU, CUDA and Vulkan
builds and consumes F32, F16, Q8_0 and Q4 GGUF files. The main advantages are
portable backends, one-patch reproducibility, bounded-memory model streaming,
and complete raw image-to-textured-GLB evidence with an independent official
reference. Vulkan owns neural inference; CUDA owns all PBR post-processing.

The CUDA build also provides a native image-to-textured-PBR-GLB path. It runs
MoGe preprocessing, GGML condition/SS/SLat/decoder stages, FlexiCubes, VTK and
MeshFix cleanup, xatlas UVs, 100 Gaussian observations, 2500 Adam/TV texture
steps and Telea repair in C++. No cuDNN, Python or Torch runtime is loaded by
this command. The official Python environment remains the independent oracle
for parity and render comparisons; BF16/SDPA versus GGUF F16/Q8/Q4 arithmetic
is reported as measured error rather than described as bitwise equality.

### Measured end-to-end snapshot

The [current complete measurement table](cpp_ggml/benchmarks/e2e_comparison/full_glb_current/README.md)
contains the actual official Python and six native F16/Q8_0/Q4_0 GLBs,
standalone 1024px PNGs, poses and 60-view final-asset comparisons. Every row
starts from the kidsroom image and mask 14, seed 42, on an RTX 3060 12 GiB.
The timer includes process startup, model loading, inference, complete
100-view/2500-step PBR and GLB/PNG output. External scoring is excluded.
Vulkan rows include their separate CUDA post-processing process.

| Pipeline | Complete cold GLB (s) | Final GLB RGB MAE | Silhouette IoU |
| --- | ---: | ---: | ---: |
| PyTorch staged mixed | 234.303 | reference | reference |
| CUDA F16 | 252.136 | 0.02626 | 0.97719 |
| CUDA Q8_0 | 216.164 | 0.02810 | 0.97386 |
| CUDA Q4_0 | 215.338 | 0.04803 | 0.92415 |
| Vulkan F16 + CUDA PBR | 236.220 | 0.02566 | 0.97676 |
| Vulkan Q8_0 + CUDA PBR | 223.370 | 0.02732 | 0.97504 |
| Vulkan Q4_0 + CUDA PBR | 224.024 | 0.05115 | 0.91703 |

The official row was remeasured after fixing unused-weight residency before
FlexiCubes; the six native raw rows are unchanged. Q8/Q4 are faster than this
official cold process, while both F16 rows are slower. All native rows exceed
70 seconds. Vulkan Q4 is not faster than Q8 in this single-sample matrix.
The independently captured official stage reference differs from the timed
official output by RGB MAE `0.00573`, IoU `0.99913`; reference agreement is
measured separately, not assumed exact.

![Complete image-to-textured-GLB latency](cpp_ggml/benchmarks/e2e_comparison/e2e_latency_current.png)

Only one complete run is recorded per row. A successful export is not an
accuracy pass or evidence of stable sub-70-second performance. The
[release gate](cpp_ggml/benchmarks/e2e_comparison/e2e_gate_current.json)
records remaining numerical, timing, repeatability and acceptance-coverage
gaps; official hot-session and native cold-process numbers are not mixed.

The release audit corrected a missing export-time UV V flip and the native
material's metallic factor. With identical official intermediate inputs, the
corrected native GLB has bitwise-identical mesh/UVs and a measured 60-view
RGB MAE of `0.001192`, IoU `1.0`. That is a controlled post-processing result,
not a claim that raw neural inference is identical. Fresh raw outputs after
the fix, current charts and the remaining-issues analysis are linked from
[benchmarks](cpp_ggml/benchmarks/README.md).

### Model families and practical advantages

The GGUF directory contains the complete stage matrix, so users can choose a
single precision policy without changing the graph:

| GGUF family | Pipeline role | Practical advantage |
| --- | --- | --- |
| `ss_generator-*`, `ss_decoder-*` | sparse structure and occupancy | controls geometry support; F16 is the native floating-point baseline, Q8_0 reduces weight storage, and Q4 is the compact experimental path |
| `slat_generator-*` | structured latent diffusion | keeps the long latent denoising stage in the same native GGML graph across CPU, CUDA and Vulkan |
| `slat_decoder_gs-*`, `slat_decoder_gs_4-*` | Gaussian attribute decoding | produces the Gaussian representation used by the official camera renderer |
| `slat_decoder_mesh-*` | mesh decoding | native FlexiCubes mesh feeds C++ cleanup, xatlas UVs and CUDA texture baking |

Every family is available in the repository's F32/F16/Q8_0 and Q4 variants
where conversion supports that tensor layout. The runtime keeps the model
files under one canonical directory, streams stages within a bounded memory
budget, and records hashes in E2E reports so a benchmark result can be
reproduced rather than inferred from a module microbenchmark.

### GGUF downloads

All model artifacts are kept in [`cpp_ggml/models/gguf/`](cpp_ggml/models/gguf/).
The SAM 3D model helper downloads the matching files from
[Asher-1/SAM_3D_OBJECTS_GGUF](https://huggingface.co/Asher-1/SAM_3D_OBJECTS_GGUF).
The requested public GGUF download index is also available at
[Asher-1/lingbot-map-gguf](https://huggingface.co/Asher-1/lingbot-map-gguf/tree/main);
it is a separate model family and must not be substituted for SAM 3D weights.
Use the repository helper to place SAM 3D files in the canonical directory:

```bash
bash cpp_ggml/scripts/download_gguf.sh --dtype q8_0
SAM3D_PYTHON=/path/to/sam3d-objects/bin/python \
  bash cpp_ggml/scripts/prepare_moge_gguf.sh
```

For the complete clone, environment, build, inference and regression workflow,
see [`cpp_ggml/README.md`](cpp_ggml/README.md) and run
`bash cpp_ggml/scripts/quickstart.sh bootstrap`.

## License

The SAM 3D Objects model checkpoints and code are licensed under [SAM License](./LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The SAM 3D Objects project was made possible with the help of many contributors.

Robbie Adkins,
Paris Baptiste,
Karen Bergan,
Kai Brown,
Michelle Chan,
Ida Cheng,
Khadijat Durojaiye,
Patrick Edwards,
Daniella Factor,
Facundo Figueroa,
Rene  de la Fuente,
Eva Galper,
Cem Gokmen,
Alex He,
Enmanuel Hernandez,
Dex Honsa,
Leonna Jones,
Arpit Kalla,
Kris Kitani,
Helen Klein,
Kei Koyama,
Robert Kuo,
Vivian Lee,
Alex Lende,
Jonny Li,
Kehan Lyu,
Faye Ma,
Mallika Malhotra,
Sasha Mitts,
William Ngan,
George Orlin,
Peter Park,
Don Pinkus,
Roman Radle,
Nikhila Ravi,
Azita Shokrpour,
Jasmine Shone,
Zayida Suber,
Phillip Thomas,
Tatum Turner,
Joseph Walker,
Meng Wang,
Claudette Ward,
Andrew Westbury,
Lea Wilken,
Nan Yang,
Yael Yungster


## Citing SAM 3D Objects

If you use SAM 3D Objects in your research, please use the following BibTeX entry.

```
@article{sam3dteam2025sam3d3dfyimages,
      title={SAM 3D: 3Dfy Anything in Images}, 
      author={SAM 3D Team and Xingyu Chen and Fu-Jen Chu and Pierre Gleize and Kevin J Liang and Alexander Sax and Hao Tang and Weiyao Wang and Michelle Guo and Thibaut Hardin and Xiang Li and Aohan Lin and Jiawei Liu and Ziqi Ma and Anushka Sagar and Bowen Song and Xiaodong Wang and Jianing Yang and Bowen Zhang and Piotr Dollár and Georgia Gkioxari and Matt Feiszli and Jitendra Malik},
      year={2025},
      eprint={2511.16624},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16624}, 
}
```
