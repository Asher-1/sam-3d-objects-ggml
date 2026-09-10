#!/usr/bin/env bash
# Shared root launcher. Native reconstruction itself runs only C++ binaries.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CPP="${ROOT}/cpp_ggml"
runner="$1"
shift

usage() {
    cat <<'EOF'
Usage: bash run_python.sh [options]
       bash run_ggml.sh [options]

  --image PATH          Input RGB/RGBA image (default: repository kidsroom image)
  --mask PATH           Binary object mask (default: kidsroom mask 14)
  --mask-dir PATH       Multi-object scene mode, fully native: reconstruct
                        every '<idx>.png' under this directory, compose the
                        scene with the official make_scene pose semantics
                        (native scene-assemble command) and render the orbit
                        (PNG frames; scene.gif with ffmpeg). Mutually
                        exclusive with --mask; no PBR baking.
  --mask-indices LIST   Optional comma-separated subset for --mask-dir (e.g. 14,17)
  --out-dir PATH        New or empty output directory (default: output/<runner>-<timestamp>)
  --seed N              Random seed, 0..2147483647 (default: 42)
  --threads N           CPU/build threads, 1..6 (default: 6)
  --backend cuda|vulkan GGML inference backend (default: cuda); PBR always uses CUDA
  --dtype f16|q8_0|q4_0 GGUF precision (default: q8_0)
  --models-dir PATH     GGUF directory (default: cpp_ggml/models/gguf)
  --moge-model PATH     MoGe GGUF (default: <models-dir>/moge_vitl-f16.gguf)
  --python PATH         Official Python interpreter; also accepts SAM3D_PYTHON
  --config PATH         Official pipeline YAML (default: checkpoints/hf/pipeline.yaml)
  --skip-build          Use existing GGML binaries
  --setup               Install the official reference environment first (Python only)
  --accept-pbr-licenses Enable existing nvdiffrast/MeshFix dependencies for native builds
  -h, --help            Show this help

Both paths run full mesh cleanup, UV, 100-view observations, 2500-step texture
baking and GLB export. They also save the base-color PNG, pose JSON (official
schema; native extras under a "native" block) and a full-E2E timing report.
GGML defaults to strict SS attention, matching the raw regression settings.
The Python path streams the official modules with the documented mixed
precision policy to fit a 12 GiB GPU.

First run on a clean clone (GGML path):
  bash cpp_ggml/scripts/download_gguf.sh --dtype q8_0   # GGUF weights
  bash cpp_ggml/scripts/prepare_moge_gguf.sh --dtype f16 # MoGe weights
  bash run_ggml.sh                                       # build + reconstruct
Python path: bash run_python.sh --setup (once), then bash run_python.sh
EOF
}
die() { printf 'error: %b\n' "$*" >&2; exit 2; }
image="${SAM3D_IMAGE:-${ROOT}/notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png}"
mask="${SAM3D_MASK:-${ROOT}/notebook/images/shutterstock_stylish_kidsroom_1640806567/14.png}"
mask_dir="${SAM3D_MASK_DIR:-}"
mask_indices="${SAM3D_MASK_INDICES:-}"
mask_given=0
out="${SAM3D_WORK_DIR:-}"
seed="${SAM3D_SEED:-42}"
threads="${SAM3D_BUILD_JOBS:-6}"
backend=cuda
dtype="${SAM3D_DTYPE:-q8_0}"
models="${SAM3D_MODELS_DIR:-${CPP}/models/gguf}"
moge="${SAM3D_MOGE_MODEL:-}"
python="${SAM3D_PYTHON:-}"
config="${ROOT}/checkpoints/hf/pipeline.yaml"
skip_build=0
setup=0
licenses="${SAM3D_ACCEPT_PBR_LICENSES:-0}"
while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --skip-build) skip_build=1; shift ;;
        --setup) setup=1; shift ;;
        --accept-pbr-licenses) licenses=1; shift ;;
        --image|--mask|--mask-dir|--mask-indices|--out-dir|--seed|--threads|--backend|--dtype|--models-dir|--moge-model|--python|--config)
            (( $# >= 2 )) || die "missing value for $1"
            [[ -n "$2" && "$2" != --* ]] || die "missing value for $1"
            case "$1" in
                --image) image="$2" ;; --mask) mask="$2"; mask_given=1 ;;
                --mask-dir) mask_dir="$2" ;; --mask-indices) mask_indices="$2" ;;
                --out-dir) out="$2" ;;
                --seed) seed="$2" ;; --threads) threads="$2" ;; --backend) backend="$2" ;;
                --dtype) dtype="$2" ;; --models-dir) models="$2" ;; --moge-model) moge="$2" ;;
                --python) python="$2" ;; --config) config="$2" ;;
            esac
            shift 2 ;;
        *) die "unknown option: $1" ;;
    esac
done
[[ "$seed" =~ ^[0-9]+$ ]] || die "seed must be a non-negative integer"
[[ ${#seed} -le 10 ]] && (( 10#$seed <= 2147483647 )) || die "seed exceeds the native signed 32-bit CLI range"
seed="$((10#$seed))"
[[ "$threads" =~ ^[1-6]$ ]] || die "threads must be in 1..6"
[[ "$backend" == cuda || "$backend" == vulkan ]] || die "backend must be cuda or vulkan"
[[ "$dtype" == f16 || "$dtype" == q8_0 || "$dtype" == q4_0 ]] || die "dtype must be f16, q8_0 or q4_0"
# Scene mode (mask directory) and single-object mode (--mask) are mutually
# exclusive; the scene composes N single-object reconstructions with the
# official make_scene pose application instead of PBR baking.
if [[ -n "$mask_dir" ]]; then
    (( mask_given == 0 )) || die "--mask-dir and --mask are mutually exclusive"
    [[ -d "$mask_dir" ]] || die "mask directory is missing: $mask_dir"
    mask_dir="$(realpath -- "$mask_dir")"
    [[ -f "$image" ]] || die "image is missing: $image"
else
    [[ -f "$image" && -f "$mask" ]] || die "image or mask is missing"
    mask="$(realpath -- "$mask")"
fi
# Resolve user-relative paths before the inference process changes directory.
image="$(realpath -- "$image")"
out="$(realpath -m -- "${out:-${ROOT}/output/${runner}-${backend}-$(date +%Y%m%d-%H%M%S)}")"
[[ ! -e "$out" || ( -d "$out" && -z "$(find "$out" -mindepth 1 -print -quit)" ) ]] ||
    die "output directory must be new or empty: $out"

# Fail fast on missing build tools instead of deep inside cmake.
if (( ! skip_build )); then
    command -v cmake >/dev/null || die "cmake not found; install cmake >= 3.24 (e.g. pip install cmake or apt install cmake)"
    if [[ "$backend" == cuda ]]; then
        command -v nvcc >/dev/null || { [[ -n "${CUDACXX:-}" ]] || die "nvcc not found; install the CUDA toolkit or set CUDACXX=/usr/local/cuda-12.x/bin/nvcc"; }
    fi
    if [[ "$backend" == vulkan ]]; then
        command -v glslc >/dev/null || die "glslc not found; install the Vulkan SDK (https://vulkan.lunarg.com)"
    fi
fi

find_official_python() {
    local conda_executable="${SAM3D_CONDA_EXE:-$(command -v conda || true)}"
    if [[ -z "$python" && -n "$conda_executable" ]]; then
        python="$($conda_executable info --base)/envs/sam3d-objects/bin/python"
    fi
    if [[ -z "$python" && -n "${CONDA_PREFIX:-}" ]]; then python="${CONDA_PREFIX}/bin/python"; fi
    if [[ -z "$python" && -n "${VIRTUAL_ENV:-}" ]]; then python="${VIRTUAL_ENV}/bin/python"; fi
    [[ -x "$python" ]]
}

if [[ "$runner" == python ]]; then
    [[ -f "$config" ]] || die "missing checkpoint configuration: $config"
    config="$(realpath -- "$config")"
    if (( setup )); then bash "${CPP}/scripts/setup_pt_ref_env.sh"; fi
    find_official_python || die "no official Python environment found. Options:\n  - bash run_python.sh --setup   (creates the sam3d-objects conda env)\n  - bash run_python.sh --python /path/to/python\n  - activate a conda/venv with the official dependencies first"
    if [[ -n "$mask_dir" ]]; then
        exec "$python" "${CPP}/scripts/run_scene_pipeline.py" \
            --runner python --image "$image" --mask-dir "$mask_dir" \
            --config "$config" --out-dir "$out" --seed "$seed" \
            ${mask_indices:+--mask-indices "$mask_indices"}
    fi
    exec "$python" "${CPP}/scripts/bench_full_e2e_pipeline.py" \
        --image "$image" --mask "$mask" --config "$config" --out-dir "$out" \
        --seed "$seed" --warmup 0 --iters 1 --stream-weights
fi
[[ "$runner" == ggml ]] || die "unsupported runner: $runner"
(( setup == 0 )) || die "--setup applies only to the Python environment"
models="$(realpath -m -- "$models")"
moge="$(realpath -m -- "${moge:-${models}/moge_vitl-f16.gguf}")"
missing_gguf=""
for stage in ss_generator ss_decoder slat_generator slat_decoder_gs slat_decoder_mesh; do
    [[ -s "${models}/${stage}-${dtype}.gguf" ]] || missing_gguf="${missing_gguf} ${stage}-${dtype}.gguf"
done
if [[ -n "$missing_gguf" ]]; then
    die "missing GGUF:${missing_gguf}\n  fix: bash cpp_ggml/scripts/download_gguf.sh --dtype ${dtype}\n  (existing files are preserved; safe to re-run)"
fi
[[ -s "$moge" ]] || die "missing MoGe GGUF: $moge\n  fix: bash cpp_ggml/scripts/prepare_moge_gguf.sh --dtype f16\n  (requires a Python env with torch; see --setup for the reference env)"
cuda_binary="${CPP}/build-cuda-pbr/bin/sam3d-cli"
vulkan_binary="${CPP}/build-vulkan/bin/sam3d-cli"
if (( ! skip_build )); then
    # nvdiffrast is non-commercial and MeshFix is GPL/commercial: the build
    # embeds both. Accept explicitly via flag/env, or interactively once.
    if (( licenses == 0 )); then
        if [[ -t 0 ]]; then
            echo "The native PBR build embeds nvdiffrast (non-commercial license)"
            echo "and MeshFix (GPL/commercial dual license). Continue? [y/N]"
            read -r reply
            [[ "$reply" == y || "$reply" == Y ]] || die "aborted; re-run with --accept-pbr-licenses to skip this prompt"
            licenses=1
        else
            die "build requires --accept-pbr-licenses (nvdiffrast non-commercial, MeshFix GPL/commercial)"
        fi
    fi
    cmake -S "$CPP" -B "${CPP}/build-cuda-pbr" -DCMAKE_BUILD_TYPE=Release \
        -DSAM3D_GGML_CUDA=ON -DSAM3D_GGML_VULKAN=OFF -DSAM3D_GGML_NATIVE_PBR=ON \
        -DSAM3D_GGML_NVDIFFRAST_NONCOMMERCIAL=ON -DSAM3D_GGML_MESHFIX_GPL=ON
    cmake --build "${CPP}/build-cuda-pbr" --target sam3d-cli --parallel "$threads"
    if [[ "$backend" == vulkan ]]; then
        cmake -S "$CPP" -B "${CPP}/build-vulkan" -DCMAKE_BUILD_TYPE=Release \
            -DSAM3D_GGML_CUDA=OFF -DSAM3D_GGML_VULKAN=ON -DSAM3D_GGML_NATIVE_PBR=OFF \
            -DSAM3D_GGML_NVDIFFRAST_NONCOMMERCIAL=OFF -DSAM3D_GGML_MESHFIX_GPL=OFF
        cmake --build "${CPP}/build-vulkan" --target sam3d-cli --parallel "$threads"
    fi
fi
[[ -x "$cuda_binary" ]] || die "missing CUDA PBR executable: $cuda_binary"
[[ "$backend" != vulkan || -x "$vulkan_binary" ]] || die "missing Vulkan executable: $vulkan_binary"
mkdir -p -- "$out"
exec > >(tee "${out}/run.log") 2>&1
trap 'echo "Reconstruction failed at line ${LINENO}; partial artifacts: ${out}" >&2' ERR
if [[ -n "$mask_dir" ]]; then
    # Multi-object scene mode, fully native: per-mask image-to-3d (no PBR),
    # then the pure-C++ scene-assemble command composes the official pose
    # semantics, normalizes, and renders the orbit with the same CUDA
    # Gaussian rasterizer that the 100-view bake uses. The official Python
    # notebook flow stays available in run_scene_pipeline.py as a reference.
    mkdir -p "${out}/objects"
    : > "${out}/scene_objects.txt"
    scene_indices=()
    if [[ -n "$mask_indices" ]]; then
        IFS=',' read -ra scene_indices <<< "$mask_indices"
    else
        local_index=0
        while [[ -f "${mask_dir}/${local_index}.png" ]]; do
            scene_indices+=("$local_index")
            local_index=$((local_index + 1))
        done
    fi
    (( ${#scene_indices[@]} > 0 )) || die "no masks found under ${mask_dir}"
    cli="$cuda_binary"
    if [[ "$backend" == vulkan ]]; then
        cli="$vulkan_binary"
        "$cuda_binary" rng-dump --implementation cuda --seed "$seed" --sizes 1 \
            --out-dir "${out}/rng_contract"
        blocks="$(sed -nE 's/.*"distribution_blocks":[[:space:]]*([0-9]+).*/\1/p' "${out}/rng_contract/rng_contract.json")"
        [[ "$blocks" =~ ^[1-9][0-9]*$ ]] || die "invalid CUDA Philox contract"
    fi
    # Session batch: one process serves every mask with the shared model
    # resources (mmap weight reload, one backend for all stages, RGB-only
    # MoGe point map reuse). The batch manifest records input/output digests
    # per object; re-running the same out-dir skips only objects whose
    # digests still verify.
    mask_list="${out}/batch_mask_list.txt"
    : > "$mask_list"
    for scene_index in "${scene_indices[@]}"; do
        mask_file="${mask_dir}/${scene_index}.png"
        [[ -f "$mask_file" ]] || die "missing mask: $mask_file"
        printf '%s\t%s\n' "$scene_index" "$mask_file" >> "$mask_list"
    done
    scene_args=(image-to-3d --model "$models" --moge-model "$moge"
        --image "$image"
        --mask-list "$mask_list" --out-dir "$out"
        --backend "$backend" --dtype "$dtype" --seed "$seed" --threads "$threads"
        # Scene preset: F16-KV flash attention. Free-run pose parity is
        # attention-mode independent after the MoGe strict-KV repair
        # (measured A/B on masks 14 and 15), and this path halves the
        # SS-flow time. The single-object pipeline below keeps the
        # strict F32-KV contract for its per-step velocity gate.
        --ss-attention normal)
    if [[ "$backend" == vulkan ]]; then
        scene_args+=(--philox-blocks "$blocks")
    fi
    "$cli" "${scene_args[@]}"
    for scene_index in "${scene_indices[@]}"; do
        # The CLI writes obj_<ID> without zero padding.
        objdir="${out}/objects/obj_${scene_index}"
        if [[ -f "${objdir}/output.ply" && -f "${objdir}/pose.json" ]]; then
            printf '%s\t%s\n' "${objdir}/output.ply" "${objdir}/pose.json" >> "${out}/scene_objects.txt"
        else
            echo "WARNING: obj_${scene_index} missing outputs; excluded from the scene" >&2
        fi
    done
    "$cuda_binary" scene-assemble --objects-list "${out}/scene_objects.txt" \
        --out-dir "$out" --num-frames 300 --radius 1.0 --fov 60 --resolution 512
    if command -v ffmpeg >/dev/null; then
        ffmpeg -loglevel error -y -framerate 30 -i "${out}/frames/frame_%04d.png" \
            "${out}/scene.gif"
        echo "Scene orbit GIF: ${out}/scene.gif"
    else
        echo "Scene orbit frames: ${out}/frames/frame_*.png (install ffmpeg for scene.gif)"
    fi
    echo "Posed scene PLY: ${out}/scene_posed.ply"
    echo "Scene manifest: ${out}/scene_manifest.json"
    exit 0
fi
command=(image-to-3d --model "$models" --moge-model "$moge" --image "$image" --mask "$mask"
    --backend "$backend" --dtype "$dtype" --seed "$seed" --threads "$threads"
    --ss-attention strict --out "${out}/output.ply" --pose-out "${out}/pose.json"
    --dtype-contract-out "${out}/dtype_contract.json")
if [[ "$backend" == cuda ]]; then
    started_ms="$(date +%s%3N)"
    "$cuda_binary" "${command[@]}" --pbr-out "${out}/output.glb"
    elapsed_ms=$(( $(date +%s%3N) - started_ms ))
    inference_binary="$cuda_binary"
else
    # The Vulkan contract owns inference only; the cold-process span includes
    # the CUDA Philox contract dump, the handoff hashing and the CUDA PBR
    # assembly so it stays comparable with the official cold process.
    started_ms="$(date +%s%3N)"
    "$cuda_binary" rng-dump --implementation cuda --seed "$seed" --sizes 1 \
        --out-dir "${out}/rng_contract"
    blocks="$(sed -nE 's/.*"distribution_blocks":[[:space:]]*([0-9]+).*/\1/p' "${out}/rng_contract/rng_contract.json")"
    [[ "$blocks" =~ ^[1-9][0-9]*$ ]] || die "invalid CUDA Philox contract"
    "$vulkan_binary" "${command[@]}" --philox-blocks "$blocks" \
        --mesh-vertices-out "${out}/vertices.samt" --mesh-faces-out "${out}/faces.samt"
    (cd "$out" && sha256sum output.ply vertices.samt faces.samt pose.json dtype_contract.json > handoff.sha256)
    "$cuda_binary" pbr-assemble --vertices "${out}/vertices.samt" --faces "${out}/faces.samt" \
        --ply "${out}/output.ply" --out "${out}/output.glb" --seed "$seed"
    (cd "$out" && sha256sum --check handoff.sha256)
    elapsed_ms=$(( $(date +%s%3N) - started_ms ))
    inference_binary="$vulkan_binary"
fi
[[ -s "${out}/output.glb" && -s "${out}/output.base_color.png" ]] || die "missing GLB or base-color texture"

# Mirror the official full-E2E receipt so run_ggml.sh and run_python.sh
# artifacts stay directly comparable (publish_full_e2e.py consumes the same
# latency_ms structure from both).
json_escape() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }
latency_ms="$(awk -v ms="$elapsed_ms" 'BEGIN { printf "%.3f", ms }')"
latency_s="$(awk -v ms="$elapsed_ms" 'BEGIN { printf "%.3f", ms / 1000.0 }')"
inference_sha="$(sha256sum -- "$inference_binary" | cut -d' ' -f1)"
pbr_sha="$(sha256sum -- "$cuda_binary" | cut -d' ' -f1)"
vulkan_report_fields=""
vulkan_artifact_fields=""
if [[ "$backend" == vulkan ]]; then
    vulkan_report_fields=$'\n  "pbr_handoff": "Vulkan inference -> CUDA PBR",'
    vulkan_artifact_fields=$',\n    "mesh_vertices": "'"$(json_escape "${out}/vertices.samt")"$'",\n    "mesh_faces": "'"$(json_escape "${out}/faces.samt")"$'",\n    "handoff_sha256": "'"$(json_escape "${out}/handoff.sha256")"$'"'
fi
cat >"${out}/native_full_e2e.json" <<EOF
{
  "schema": "sam3d.native-full-e2e.v1",
  "timer_contract": {
    "id": "sam3d.cold-process.image-mask-to-textured-pbr-glb.v1",
    "kind": "cold-process",
    "includes": [
      "process start", "GGUF weight load", "neural inference",
      "mesh cleanup", "xatlas UV", "100 Gaussian observations",
      "2500-step Adam/TV bake", "Telea", "GLB export and file close"
    ],
    "excludes": ["cmake configure and build when performed", "external comparison rendering"]
  },
  "image": "$(json_escape "$image")",
  "mask": "$(json_escape "$mask")",
  "seed": ${seed},
  "backend": "${backend}",${vulkan_report_fields}
  "pbr_backend": "cuda",
  "dtype": "${dtype}",
  "ss_attention": "strict",
  "models_dir": "$(json_escape "$models")",
  "moge_model": "$(json_escape "$moge")",
  "inference_binary": "$(json_escape "$inference_binary")",
  "inference_binary_sha256": "${inference_sha}",
  "pbr_binary": "$(json_escape "$cuda_binary")",
  "pbr_binary_sha256": "${pbr_sha}",
  "warmup": 0,
  "iters": 1,
  "latency_ms": {"mean": ${latency_ms}, "p50": ${latency_ms}, "max": ${latency_ms}},
  "latency_s": ${latency_s},
  "outputs": ["$(json_escape "${out}/output.glb")"],
  "artifacts": {
    "ply": "$(json_escape "${out}/output.ply")",
    "base_color_png": "$(json_escape "${out}/output.base_color.png")",
    "pose": "$(json_escape "${out}/pose.json")",
    "dtype_contract": "$(json_escape "${out}/dtype_contract.json")",
    "run_log": "$(json_escape "${out}/run.log")"${vulkan_artifact_fields}
  },
  "gpu_exclusivity": {"required": false, "checked": false, "active_compute_processes": []}
}
EOF
echo "Textured model: ${out}/output.glb"
echo "Base-color texture: ${out}/output.base_color.png"
echo "Full-E2E report: ${out}/native_full_e2e.json"
