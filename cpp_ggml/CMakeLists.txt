cmake_minimum_required(VERSION 3.14)
project(sam3dggml LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Options
option(SAM3D_GGML_BUILD_CLI  "Build the sam3d-cli binary"   ON)
option(SAM3D_GGML_SHARED     "Build libsam3dggml as a shared library" OFF)
option(SAM3D_GGML_BUILD_TESTS "Build lightweight integration tests" ON)

# Backend forwarding to ggml (mirrors the ultralytics-ggml pattern)
option(SAM3D_GGML_CUDA    "Enable CUDA backend in ggml"    OFF)
option(SAM3D_GGML_VULKAN  "Enable Vulkan backend in ggml"  OFF)
option(SAM3D_GGML_METAL   "Enable Metal backend in ggml"   OFF)
option(SAM3D_GGML_HIPBLAS "Enable HIPBLAS backend in ggml" OFF)

# The official textured-GLB path currently renders source Gaussians and runs
# differentiable UV texture baking with native CUDA implementations. The source
# tree has no equivalent Vulkan renderer, differentiable texture
# sampler/backward path, or Adam/TV kernels. The supported deployment topology
# is intentionally split: a Vulkan build exports neural PLY and raw FlexiCubes
# mesh, then a separate CUDA PBR build consumes those immutable artifacts. Do
# not enable native PBR in a Vulkan build or label the CUDA result Vulkan PBR.
set(_sam3d_native_pbr_default OFF)
if(SAM3D_GGML_CUDA)
    set(_sam3d_native_pbr_default ON)
endif()
option(SAM3D_GGML_NATIVE_PBR
       "Build the native CUDA Gaussian renderer and textured-GLB stages"
       ${_sam3d_native_pbr_default})
option(SAM3D_GGML_MESHFIX_GPL
       "Enable GPL-3.0 pymeshfix TMesh boundary repair for exact official output"
       OFF)
option(SAM3D_GGML_NVDIFFRAST_NONCOMMERCIAL
       "Enable NVIDIA nvdiffrast CUDA core under its non-commercial source license"
       OFF)
if(SAM3D_GGML_NATIVE_PBR AND NOT SAM3D_GGML_CUDA)
    message(FATAL_ERROR
        "SAM3D_GGML_NATIVE_PBR requires SAM3D_GGML_CUDA=ON: this tree has no "
        "Vulkan Gaussian renderer, differentiable UV sampler/backward path, or "
        "Adam/TV texture-bake implementation. Use a separate CUDA PBR build "
        "with Vulkan PLY and raw-mesh handoff artifacts.")
endif()
if(SAM3D_GGML_MESHFIX_GPL AND NOT SAM3D_GGML_NATIVE_PBR)
    message(FATAL_ERROR
        "SAM3D_GGML_MESHFIX_GPL requires SAM3D_GGML_NATIVE_PBR=ON")
endif()
if(SAM3D_GGML_NVDIFFRAST_NONCOMMERCIAL AND NOT SAM3D_GGML_NATIVE_PBR)
    message(FATAL_ERROR
        "SAM3D_GGML_NVDIFFRAST_NONCOMMERCIAL requires SAM3D_GGML_NATIVE_PBR=ON")
endif()
if(SAM3D_GGML_CUDA OR SAM3D_GGML_NATIVE_PBR)
    # ggml enables CUDA in its own subdirectory. The native renderer owns .cu
    # sources in this directory, so the top-level project must enable the
    # language as well for CMake to generate compile and archive rules.
    enable_language(CUDA)
endif()

set(_sam3d_gpu_backends 0)
foreach(_backend SAM3D_GGML_CUDA SAM3D_GGML_METAL SAM3D_GGML_VULKAN SAM3D_GGML_HIPBLAS)
    if(${_backend})
        math(EXPR _sam3d_gpu_backends "${_sam3d_gpu_backends} + 1")
    endif()
endforeach()
if(_sam3d_gpu_backends GREATER 1)
    message(FATAL_ERROR "Select at most one GPU backend per build directory")
endif()

# Keep a clean CUDA configuration reproducible across CMake versions. CMake
# otherwise leaves this empty before its first CUDA try-compile, which also
# makes unrelated feature checks (such as OpenMP) fail. An explicit caller
# selection always wins over this local attached-GPU fallback.
if(SAM3D_GGML_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        execute_process(
            COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _sam3d_cuda_compute_cap
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        string(REGEX MATCH "^[0-9]+\\.[0-9]+" _sam3d_cuda_compute_cap "${_sam3d_cuda_compute_cap}")
        if(_sam3d_cuda_compute_cap)
            string(REPLACE "." "" CMAKE_CUDA_ARCHITECTURES "${_sam3d_cuda_compute_cap}")
            message(STATUS "CUDA architecture auto-detected: ${CMAKE_CUDA_ARCHITECTURES}")
        endif()
    endif()
endif()

set(GGML_CUDA ${SAM3D_GGML_CUDA} CACHE BOOL "CUDA backend" FORCE)
set(GGML_METAL ${SAM3D_GGML_METAL} CACHE BOOL "Metal backend" FORCE)
set(GGML_VULKAN ${SAM3D_GGML_VULKAN} CACHE BOOL "Vulkan backend" FORCE)
set(GGML_HIP ${SAM3D_GGML_HIPBLAS} CACHE BOOL "HIP backend" FORCE)
if(SAM3D_GGML_CUDA)
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "Replay CUDA graphs" FORCE)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ggml tuning: llamafile tinyBLAS helps the F32 GEMM hotspots of this workload.
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE      OFF CACHE BOOL "tinyBLAS SGEMM: OFF - its F16xf32 path produces INF for our shapes (see third_party/ggml-patches note)" FORCE)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/include/ggml.h")
    message(FATAL_ERROR
        "ggml submodule missing. Run: git submodule update --init --recursive")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Apply the single local ggml patch before descending into the submodule. It
# lives in cpp_ggml/third_party/ggml-patches/ and is applied idempotently by
# cpp_ggml/scripts/apply_ggml_patches.sh; re-running CMake is a no-op once
# it is in place. NEVER edit third_party/ggml by hand — uncommitted
# submodule changes are silently lost on the next `git submodule update`.
# The submodule is pinned to ggml v0.21.0.
# ─────────────────────────────────────────────────────────────────────────────
execute_process(
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/scripts/apply_ggml_patches.sh
    RESULT_VARIABLE _sam3d_ggml_patch_result
    OUTPUT_VARIABLE _sam3d_ggml_patch_output
    ERROR_VARIABLE  _sam3d_ggml_patch_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)
if(NOT _sam3d_ggml_patch_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to apply ggml patches (exit ${_sam3d_ggml_patch_result}):\n"
        "stdout:\n${_sam3d_ggml_patch_output}\n"
        "stderr:\n${_sam3d_ggml_patch_error}")
endif()
message(STATUS "${_sam3d_ggml_patch_output}")

if(SAM3D_GGML_CUDA)
    # PyTorch 2.5.1 selects its F16 aligned CUTLASS memory-efficient attention
    # kernel for the real mesh fixture.  We compile only that public source
    # header closure, pinned by content hash; this neither links libtorch nor
    # introduces a cuDNN runtime dependency.
    include(FetchContent)
    set(SAM3D_PYTORCH_MEM_EFF_SOURCE_DIR
        "${CMAKE_BINARY_DIR}/third_party/pytorch-v2.5.1-mem-eff-attention")
    set(_sam3d_pytorch_source_url
        "https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/aten/src")
    function(sam3d_fetch_pytorch_mem_eff_header relative_path expected_sha256)
        set(destination "${SAM3D_PYTORCH_MEM_EFF_SOURCE_DIR}/${relative_path}")
        if(EXISTS "${destination}")
            file(SHA256 "${destination}" actual_sha256)
            if(actual_sha256 STREQUAL expected_sha256)
                return()
            endif()
        endif()
        get_filename_component(destination_dir "${destination}" DIRECTORY)
        file(MAKE_DIRECTORY "${destination_dir}")
        file(DOWNLOAD "${_sam3d_pytorch_source_url}/${relative_path}" "${destination}"
            EXPECTED_HASH "SHA256=${expected_sha256}" TLS_VERIFY ON
            STATUS download_status)
        list(GET download_status 0 download_code)
        list(GET download_status 1 download_message)
        if(NOT download_code EQUAL 0)
            message(FATAL_ERROR "Failed to fetch pinned PyTorch attention header ${relative_path}: ${download_message}")
        endif()
    endfunction()
    sam3d_fetch_pytorch_mem_eff_header("ATen/native/transformers/cuda/mem_eff_attention/debug_utils.h" "0bc7029c77ea42c8144e92814a7417c7b49acb4dee002fba0763a35a7ee27c02")
    sam3d_fetch_pytorch_mem_eff_header("ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_pipelined.h" "220dcb8e76c6f95339f9ef8630aae000a2c65fc003f7063d0c1d2da7df5ef8d8")
    sam3d_fetch_pytorch_mem_eff_header("ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_rescale_output.h" "01937cbd90f837ba0aa0674a111b56b7a2fdac93acd7bd46384066a26310f78c")
    sam3d_fetch_pytorch_mem_eff_header("ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_thread_apply_logsumexp.h" "c5ca7711ae4f94cd2b1c3cd8275114b5740b804748ca4d08c501bed9f42f6941")
    sam3d_fetch_pytorch_mem_eff_header("ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma.h" "5bdb4c6606d73a76fd740e762dcda90b175039d5e13cec18e1b074e1e9179397")
    sam3d_fetch_pytorch_mem_eff_header("ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma_base.h" "9281f07cebb654fb3107a4150a9b840a54a0318957ec8c4c9535586d6369b6c1")
    sam3d_fetch_pytorch_mem_eff_header("ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma_multistage.h" "2b73ea750e2f2455d42a2b6dcdf2553ed223a1efcd0865688866f3253dd80b41")
    sam3d_fetch_pytorch_mem_eff_header("ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma_pipelined.h" "71c2bb92d8a3ba5926cf27d1bf4657cd61b018f9cc6f03d937386bbb1f4876f1")
    sam3d_fetch_pytorch_mem_eff_header("ATen/native/transformers/cuda/mem_eff_attention/gemm/find_default_mma.h" "223613991c96520b6dbeeefeb160b948483f80cb60800e81f184c22a0d335994")
    sam3d_fetch_pytorch_mem_eff_header("ATen/native/transformers/cuda/mem_eff_attention/gemm/mma_accum_lambda_iterator.h" "ec666626f49dd4b6570a3f284735bf38bbbd5d56b53c49ffd3ff1f9345462784")
    sam3d_fetch_pytorch_mem_eff_header("ATen/native/transformers/cuda/mem_eff_attention/gemm/mma_from_smem.h" "aa18d0eb6812037d128471464a20887192c84193e8f78396b8d3b1a3442abce8")
    sam3d_fetch_pytorch_mem_eff_header("ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h" "ce914b76b79d92c487a44cfc3ed19f41da947d0354672ab716489091fb0a7cbf")
    sam3d_fetch_pytorch_mem_eff_header("ATen/native/transformers/cuda/mem_eff_attention/iterators/default_warp_iterator_from_smem.h" "c46d8c01507a4a049f784545379b323194750f9b34b3400a020f2ccc2bb912d8")
    sam3d_fetch_pytorch_mem_eff_header("ATen/native/transformers/cuda/mem_eff_attention/iterators/epilogue_predicated_tile_iterator.h" "bee2b0cf715346a45c96a45a16bbe7f362a2a74d2ec5b1cd4410881a3920fd4f")
    sam3d_fetch_pytorch_mem_eff_header("ATen/native/transformers/cuda/mem_eff_attention/iterators/make_residual_last.h" "17f4e1b6e4b473e21e106d2d4f71a07196a6849d41be8769c16fd2db43a0e666")
    sam3d_fetch_pytorch_mem_eff_header("ATen/native/transformers/cuda/mem_eff_attention/iterators/predicated_tile_access_iterator_residual_last.h" "d17ec394229e52e693cea853694f0278f813624dc7cd497e2a2ace515d1d9ff6")
    sam3d_fetch_pytorch_mem_eff_header("ATen/native/transformers/cuda/mem_eff_attention/iterators/predicated_tile_iterator_residual_last.h" "880a3383d716b50f5527564d01e066a8eabeace93a534562065c55fa4faaf220")
    sam3d_fetch_pytorch_mem_eff_header("ATen/native/transformers/cuda/mem_eff_attention/iterators/transpose_warp_iterator.h" "8eb466e7e3013da32d0a083a3e5df978d7983b83a2dd2a5e362c526892ae894e")
    sam3d_fetch_pytorch_mem_eff_header("ATen/native/transformers/cuda/mem_eff_attention/iterators/warp_iterator_from_smem.h" "3e97f7b3eae7cb5c21fb9b18761a6472b0b1b317923508c6a0cfcf8240983258")
    sam3d_fetch_pytorch_mem_eff_header("ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h" "6019c38609117f0b7c28c7f2da4cbd24713aaa3d37b028d23da744bfd46dd2ec")
    sam3d_fetch_pytorch_mem_eff_header("ATen/native/transformers/cuda/mem_eff_attention/transform/tile_smem_loader.h" "798f55c4a47fadbaad11316d7a175f1063f63919069e034cf67debc946a2ecb0")

    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_Declare(sam3d_cutlass
        URL "https://github.com/NVIDIA/cutlass/archive/bbe579a9e3beb6ea6626d9227ec32d0dae119a49.tar.gz"
        URL_HASH "SHA256=9fa1da6be3d2d9207b801d5768cbced59c202444a8c84b82325b0670f47f9d48"
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
    FetchContent_GetProperties(sam3d_cutlass)
    if(NOT sam3d_cutlass_POPULATED)
        FetchContent_Populate(sam3d_cutlass)
    endif()
endif()

add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

if(SAM3D_GGML_CUDA)
    # spconv 2.3.8's selected Ampere sparse F16 kernel is vendored as
    # generated Cumm source.  It is independent of cuDNN and is linked only
    # into CUDA builds; Vulkan/CPU graphs keep their existing portable path.
    file(GLOB_RECURSE SAM3D_SPCONV_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sam3d-spconv/src/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sam3d-spconv/src/*.cc")
    add_library(sam3d_spconv STATIC ${SAM3D_SPCONV_SOURCES})
    target_include_directories(sam3d_spconv SYSTEM PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sam3d-spconv/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sam3d-spconv/cumm-include")
    target_link_libraries(sam3d_spconv PUBLIC CUDA::cudart)
    set_target_properties(sam3d_spconv PROPERTIES
        CXX_STANDARD 14 CUDA_STANDARD 14 CUDA_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(sam3d_spconv PRIVATE TV_ENABLE_HARDWARE_ACC)
    target_compile_options(sam3d_spconv PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr
            -Xcudafe=--diag_suppress=implicit_return_from_non_void_function>)

    target_include_directories(ggml-cuda PRIVATE
        "${SAM3D_PYTORCH_MEM_EFF_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pytorch-mem-eff-attention-compat"
        "${sam3d_cutlass_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sam3d-spconv/include")
    target_link_libraries(ggml-cuda PRIVATE sam3d_spconv)
    target_compile_options(ggml-cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

# stb is vendored by the pinned ggml source.  asset_io.cpp owns the single
# implementation translation unit; this target exposes only its headers.
add_library(sam3d_stb INTERFACE)
target_include_directories(sam3d_stb INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/examples)

# Native counterpart of the official Python xatlas.parametrize call. The
# vendored source is an unmodified MIT-licensed xatlas snapshot; see its
# headers for the complete license text and provenance.
find_package(Threads REQUIRED)
find_package(JPEG REQUIRED)
add_library(sam3d_xatlas STATIC third_party/xatlas/xatlas.cpp)
target_include_directories(sam3d_xatlas PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xatlas)
target_link_libraries(sam3d_xatlas PUBLIC Threads::Threads)

if(SAM3D_GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(SAM3D_GGML_NATIVE_PBR)
    find_package(VTK 9.3 CONFIG REQUIRED COMPONENTS
        CommonCore CommonDataModel CommonExecutionModel FiltersCore)
    find_package(OpenCV 4.2 REQUIRED COMPONENTS core photo)
    set(SAM3D_MIP_GAUSSIAN_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mip-splatting-diff-gaussian-rasterization)
    if(NOT EXISTS ${SAM3D_MIP_GAUSSIAN_DIR}/cuda_rasterizer/rasterizer_impl.cu)
        message(FATAL_ERROR
            "Missing native Gaussian rasterizer. Restore "
            "third_party/mip-splatting-diff-gaussian-rasterization from this checkout.")
    endif()
    # These upstream CUDA files are compiled as-is except for the single
    # gsplat-parity delta recorded in the fork's UPSTREAM.md ("Local deltas").
    # The wrapper in src/gaussian_renderer.cu owns SAM3D-specific file I/O and
    # camera construction; it never changes renderer math.
    add_library(sam3d_mip_gaussian_rasterizer STATIC
        ${SAM3D_MIP_GAUSSIAN_DIR}/cuda_rasterizer/backward.cu
        ${SAM3D_MIP_GAUSSIAN_DIR}/cuda_rasterizer/forward.cu
        ${SAM3D_MIP_GAUSSIAN_DIR}/cuda_rasterizer/rasterizer_impl.cu)
    target_include_directories(sam3d_mip_gaussian_rasterizer
        PUBLIC ${SAM3D_MIP_GAUSSIAN_DIR}/cuda_rasterizer
        PRIVATE ${SAM3D_MIP_GAUSSIAN_DIR}/third_party/glm)
    target_link_libraries(sam3d_mip_gaussian_rasterizer PUBLIC CUDA::cudart)
    set_target_properties(sam3d_mip_gaussian_rasterizer PROPERTIES
        CXX_STANDARD 17 CUDA_STANDARD 17 CUDA_EXTENSIONS OFF)

    # nvdiffrast is the official optimized UV/face rasterizer.  Its license is
    # non-commercial, so it is never fetched or linked unless the caller opts
    # in explicitly.  The selected subset is the CUDA raster core only; our
    # wrapper never includes Python, PyTorch, or cuDNN.
    if(SAM3D_GGML_NVDIFFRAST_NONCOMMERCIAL)
        include(FetchContent)
        if(POLICY CMP0169)
            cmake_policy(SET CMP0169 OLD)
        endif()
        FetchContent_Declare(sam3d_nvdiffrast_source
            GIT_REPOSITORY https://github.com/NVlabs/nvdiffrast.git
            GIT_TAG 253ac4fcea7de5f396371124af597e6cc957bfae
            GIT_SHALLOW FALSE)
        FetchContent_GetProperties(sam3d_nvdiffrast_source)
        if(NOT sam3d_nvdiffrast_source_POPULATED)
            FetchContent_Populate(sam3d_nvdiffrast_source)
        endif()
        set(_sam3d_nvdiffrast_dir ${sam3d_nvdiffrast_source_SOURCE_DIR}/csrc/common)
        add_library(sam3d_nvdiffrast_cudaraster STATIC
            src/nvdiffrast_buffer.cpp
            src/nvdiffrast_cuda_raster.cpp
            src/nvdiffrast_raster_impl.cpp
            src/nvdiffrast_raster_kernel.cu
            # The official optimized texture bake uses these CUDA-core
            # components in addition to CudaRaster.  They are compiled from
            # the same pinned nvdiffrast revision, without its Torch binding.
            src/nvdiffrast_rasterize_shader.cu
            src/nvdiffrast_interpolate_shader.cu
            src/nvdiffrast_texture.cpp
            src/nvdiffrast_texture_kernel.cu
            src/nvdiffrast_common.cpp)
        target_include_directories(sam3d_nvdiffrast_cudaraster
            PUBLIC ${_sam3d_nvdiffrast_dir}/cudaraster
            PRIVATE ${sam3d_nvdiffrast_source_SOURCE_DIR})
        target_link_libraries(sam3d_nvdiffrast_cudaraster PUBLIC CUDA::cudart)
        set_target_properties(sam3d_nvdiffrast_cudaraster PROPERTIES
            CXX_STANDARD 17 CUDA_STANDARD 17 CUDA_EXTENSIONS OFF)
    endif()

    # The official postprocess invokes `pymeshfix.PyTMesh` directly. Its
    # underlying TMesh source is GPL-3.0-or-commercial, so exact boundary
    # repair is a deliberate opt-in and never enters the default runtime.
    if(SAM3D_GGML_MESHFIX_GPL)
        include(FetchContent)
        # We only consume the upstream source files below. Direct populate is
        # intentional here: MakeAvailable() would configure pymeshfix's Python
        # extension and introduce an unrelated Python build dependency.
        if(POLICY CMP0169)
            cmake_policy(SET CMP0169 OLD)
        endif()
        FetchContent_Declare(sam3d_meshfix_source
            GIT_REPOSITORY https://github.com/pyvista/pymeshfix.git
            GIT_TAG 79adef9dd54e7f2c5fa9ba5b8bdb378a80946cea
            GIT_SHALLOW FALSE)
        FetchContent_GetProperties(sam3d_meshfix_source)
        if(NOT sam3d_meshfix_source_POPULATED)
            FetchContent_Populate(sam3d_meshfix_source)
        endif()
        set(_sam3d_meshfix_dir ${sam3d_meshfix_source_SOURCE_DIR}/src)
        add_library(sam3d_meshfix_gpl STATIC
            ${_sam3d_meshfix_dir}/checkAndRepair.cpp
            ${_sam3d_meshfix_dir}/coordinates.cpp
            ${_sam3d_meshfix_dir}/detectIntersections.cpp
            ${_sam3d_meshfix_dir}/edge.cpp
            ${_sam3d_meshfix_dir}/graph.cpp
            ${_sam3d_meshfix_dir}/heap.cpp
            ${_sam3d_meshfix_dir}/holeFilling.cpp
            ${_sam3d_meshfix_dir}/io.cpp
            ${_sam3d_meshfix_dir}/jqsort.cpp
            ${_sam3d_meshfix_dir}/list.cpp
            ${_sam3d_meshfix_dir}/marchIntersections.cpp
            ${_sam3d_meshfix_dir}/matrix.cpp
            ${_sam3d_meshfix_dir}/orientation.c
            ${_sam3d_meshfix_dir}/point.cpp
            ${_sam3d_meshfix_dir}/tin.cpp
            ${_sam3d_meshfix_dir}/tmesh.cpp
            ${_sam3d_meshfix_dir}/triangle.cpp
            ${_sam3d_meshfix_dir}/vertex.cpp)
        target_include_directories(sam3d_meshfix_gpl PUBLIC ${_sam3d_meshfix_dir})
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            target_compile_definitions(sam3d_meshfix_gpl PUBLIC IS64BITPLATFORM)
        endif()
        set_target_properties(sam3d_meshfix_gpl PROPERTIES
            CXX_STANDARD 17 CXX_EXTENSIONS OFF)
    endif()
endif()

set(SAM3DGGML_SOURCES
    src/common.cpp
    src/asset_io.cpp
    src/image_preprocess.cpp
    src/gguf_loader.cpp
    src/backend.cpp
    src/graph_builder.cpp
    src/dino_graph.cpp
    src/moge_graph.cpp
    src/moge_inference.cpp
    src/pointpatch_graph.cpp
    src/ss_flow_graph.cpp
    src/ss_decoder_graph.cpp
    src/sparse_ops.cpp
    src/slat_flow_graph.cpp
    src/gs_decoder_graph.cpp
    src/mesh_decoder_graph.cpp
    src/mesh_decoder_runner.cpp
    src/flexicubes.cpp
    src/mesh_uv.cpp
    src/mesh_rasterizer.cpp
    src/mesh_boundary_repair.cpp
    src/pytorch_philox_rng.cpp
    src/pose_decoder.cpp
    src/dtype_contract.cpp
    src/scene_assemble.cpp
    src/session.cpp
)

if(SAM3D_GGML_NATIVE_PBR)
    list(APPEND SAM3DGGML_SOURCES
        src/mesh_postprocess.cpp
        src/texture_inpaint.cpp
        src/gaussian_renderer.cpp
        src/gaussian_renderer.cu)
    if(SAM3D_GGML_NVDIFFRAST_NONCOMMERCIAL)
        list(APPEND SAM3DGGML_SOURCES
            src/mesh_rasterizer_cuda.cu
            src/texture_baker_cuda.cu
            src/native_pbr_pipeline.cpp)
    endif()
endif()

if(SAM3D_GGML_CUDA)
    list(APPEND SAM3DGGML_SOURCES src/pytorch_cuda_rng.cu)
endif()

if(SAM3D_GGML_SHARED)
    add_library(sam3dggml SHARED ${SAM3DGGML_SOURCES})
else()
    add_library(sam3dggml STATIC ${SAM3DGGML_SOURCES})
endif()

# Propagate the active GPU backend to src/backend.cpp so it knows which
# ggml_backend_*_init() to call.
if(SAM3D_GGML_CUDA)
    target_compile_definitions(sam3dggml PUBLIC SAM3D_USE_CUDA)
    target_link_libraries(sam3dggml PRIVATE CUDA::cudart)
endif()
if(SAM3D_GGML_METAL)
    target_compile_definitions(sam3dggml PUBLIC SAM3D_USE_METAL)
endif()
if(SAM3D_GGML_VULKAN)
    target_compile_definitions(sam3dggml PUBLIC SAM3D_USE_VULKAN)
endif()
if(SAM3D_GGML_HIPBLAS)
    target_compile_definitions(sam3dggml PUBLIC SAM3D_USE_CUDA)
endif()

target_include_directories(sam3dggml
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(sam3dggml
    PUBLIC  ggml
    PRIVATE sam3d_stb sam3d_xatlas JPEG::JPEG)
if(SAM3D_GGML_NATIVE_PBR)
    target_link_libraries(sam3dggml PRIVATE sam3d_mip_gaussian_rasterizer VTK::FiltersCore
                                            opencv_core opencv_photo)
    target_compile_definitions(sam3dggml PUBLIC SAM3D_NATIVE_PBR_CUDA)
    if(SAM3D_GGML_NVDIFFRAST_NONCOMMERCIAL)
        target_link_libraries(sam3dggml PRIVATE sam3d_nvdiffrast_cudaraster CUDA::cublas)
        target_include_directories(sam3dggml PRIVATE ${sam3d_nvdiffrast_source_SOURCE_DIR})
        target_compile_definitions(sam3dggml PUBLIC SAM3D_USE_NVDIFFRAST_CUDARASTER)
    endif()
    if(SAM3D_GGML_MESHFIX_GPL)
        target_link_libraries(sam3dggml PRIVATE sam3d_meshfix_gpl)
        target_compile_definitions(sam3dggml PRIVATE SAM3D_USE_MESHFIX_GPL)
    endif()
endif()

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sam3dggml PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(sam3dggml PRIVATE SAM3D_USE_OPENMP)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sam3dggml PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# CLI
if(SAM3D_GGML_BUILD_CLI)
    add_executable(sam3d-cli src/cli.cpp)
    target_link_libraries(sam3d-cli PRIVATE sam3dggml)
    target_include_directories(sam3d-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(sam3d-cli PRIVATE -Wall -Wextra -Wno-unused-parameter)
    endif()
endif()

if(SAM3D_GGML_BUILD_TESTS)
    include(CTest)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(BUILD_TESTING)
        add_test(NAME sam3d_ggml_patch_replay
                 COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/scripts/apply_ggml_patches.sh)
        set_tests_properties(sam3d_ggml_patch_replay PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        if(SAM3D_GGML_BUILD_CLI)
            add_test(NAME sam3d_ggml_cli_help
                     COMMAND $<TARGET_FILE:sam3d-cli> --help)
        endif()
        add_executable(sam3d-matmul-backend-test src/matmul_backend_test.cpp)
        target_link_libraries(sam3d-matmul-backend-test PRIVATE sam3dggml)
        add_test(NAME sam3d_ggml_matmul_cpu COMMAND sam3d-matmul-backend-test cpu)
        if(SAM3D_GGML_VULKAN)
            add_test(NAME sam3d_ggml_matmul_vulkan COMMAND sam3d-matmul-backend-test vulkan --large)
            set_tests_properties(sam3d_ggml_matmul_vulkan PROPERTIES RESOURCE_LOCK sam3d_gpu)
        endif()
        add_executable(sam3d-scatter-mean-backend-test src/scatter_mean_backend_test.cpp)
        target_link_libraries(sam3d-scatter-mean-backend-test PRIVATE sam3dggml)
        add_test(NAME sam3d_ggml_scatter_mean_cpu COMMAND sam3d-scatter-mean-backend-test cpu)
        if(SAM3D_GGML_VULKAN)
            add_test(NAME sam3d_ggml_scatter_mean_vulkan COMMAND sam3d-scatter-mean-backend-test vulkan)
            set_tests_properties(sam3d_ggml_scatter_mean_vulkan PROPERTIES RESOURCE_LOCK sam3d_gpu)
        endif()
        if(SAM3D_GGML_CUDA)
            add_test(NAME sam3d_ggml_matmul_cuda COMMAND sam3d-matmul-backend-test cuda --large)
            set_tests_properties(sam3d_ggml_matmul_cuda PROPERTIES RESOURCE_LOCK sam3d_gpu)
        endif()
        add_executable(sam3d-asset-io-test src/asset_io_test.cpp)
        target_link_libraries(sam3d-asset-io-test PRIVATE sam3dggml)
        add_test(NAME sam3d_ggml_native_glb_asset
                 COMMAND $<TARGET_FILE:sam3d-asset-io-test>)
        add_executable(sam3d-image-preprocess-test src/image_preprocess_test.cpp)
        target_link_libraries(sam3d-image-preprocess-test PRIVATE sam3dggml)
        add_test(NAME sam3d_ggml_native_image_preprocess
                 COMMAND $<TARGET_FILE:sam3d-image-preprocess-test>)
        add_executable(sam3d-moge-inference-test src/moge_inference_test.cpp)
        target_link_libraries(sam3d-moge-inference-test PRIVATE sam3dggml)
        add_test(NAME sam3d_ggml_native_moge_inference_contract
                 COMMAND $<TARGET_FILE:sam3d-moge-inference-test>)
        add_executable(sam3d-mesh-tables-test src/mesh_tables_test.cpp)
        target_link_libraries(sam3d-mesh-tables-test PRIVATE sam3dggml)
        add_test(NAME sam3d_ggml_mesh_subdivide_tables
                 COMMAND $<TARGET_FILE:sam3d-mesh-tables-test>)
        add_executable(sam3d-sparse-tables-test src/sparse_tables_test.cpp)
        target_link_libraries(sam3d-sparse-tables-test PRIVATE sam3dggml)
        add_test(NAME sam3d_ggml_spconv_implicit_gemm_tables
                 COMMAND $<TARGET_FILE:sam3d-sparse-tables-test>)
        add_executable(sam3d-flexicubes-test src/flexicubes_test.cpp)
        target_link_libraries(sam3d-flexicubes-test PRIVATE sam3dggml)
        add_test(NAME sam3d_ggml_flexicubes_synthetic_mesh
                 COMMAND $<TARGET_FILE:sam3d-flexicubes-test>)
        add_executable(sam3d-mesh-uv-test src/mesh_uv_test.cpp)
        target_link_libraries(sam3d-mesh-uv-test PRIVATE sam3dggml)
        add_test(NAME sam3d_ggml_xatlas_uv
                 COMMAND $<TARGET_FILE:sam3d-mesh-uv-test>)
        add_executable(sam3d-mesh-rasterizer-test src/mesh_rasterizer_test.cpp)
        target_link_libraries(sam3d-mesh-rasterizer-test PRIVATE sam3dggml)
        add_test(NAME sam3d_ggml_native_mesh_rasterizer
                 COMMAND $<TARGET_FILE:sam3d-mesh-rasterizer-test>)
        add_executable(sam3d-pose-decoder-test src/pose_decoder_test.cpp)
        target_link_libraries(sam3d-pose-decoder-test PRIVATE sam3dggml)
        add_test(NAME sam3d_ggml_native_pose_decoder
                 COMMAND $<TARGET_FILE:sam3d-pose-decoder-test>)
        add_executable(sam3d-dtype-contract-test src/dtype_contract_test.cpp)
        target_link_libraries(sam3d-dtype-contract-test PRIVATE sam3dggml)
        add_test(NAME sam3d_ggml_dtype_contract
                 COMMAND $<TARGET_FILE:sam3d-dtype-contract-test>)
        if(SAM3D_GGML_NATIVE_PBR)
            add_executable(sam3d-mesh-postprocess-test src/mesh_postprocess_test.cpp)
            target_link_libraries(sam3d-mesh-postprocess-test PRIVATE sam3dggml)
            add_test(NAME sam3d_ggml_vtk_mesh_simplify
                     COMMAND $<TARGET_FILE:sam3d-mesh-postprocess-test>)
            add_executable(sam3d-texture-inpaint-test src/texture_inpaint_test.cpp)
            target_link_libraries(sam3d-texture-inpaint-test PRIVATE sam3dggml)
            add_test(NAME sam3d_ggml_native_telea_inpaint
                     COMMAND $<TARGET_FILE:sam3d-texture-inpaint-test>)
            add_executable(sam3d-gaussian-renderer-test src/gaussian_renderer_test.cpp)
            target_link_libraries(sam3d-gaussian-renderer-test PRIVATE sam3dggml)
            add_test(NAME sam3d_ggml_native_gaussian_renderer
                     COMMAND $<TARGET_FILE:sam3d-gaussian-renderer-test>)
            if(SAM3D_GGML_NVDIFFRAST_NONCOMMERCIAL)
                add_executable(sam3d-texture-baker-test src/texture_baker_test.cpp)
                target_link_libraries(sam3d-texture-baker-test PRIVATE sam3dggml)
                add_test(NAME sam3d_ggml_native_texture_baker
                         COMMAND $<TARGET_FILE:sam3d-texture-baker-test>)
                add_executable(sam3d-native-pbr-pipeline-test src/native_pbr_pipeline_test.cpp)
                target_link_libraries(sam3d-native-pbr-pipeline-test PRIVATE sam3dggml)
                add_test(NAME sam3d_ggml_native_pbr_pipeline
                         COMMAND $<TARGET_FILE:sam3d-native-pbr-pipeline-test>)
            endif()
            if(SAM3D_GGML_MESHFIX_GPL)
                add_executable(sam3d-mesh-boundary-repair-test src/mesh_boundary_repair_test.cpp)
                target_link_libraries(sam3d-mesh-boundary-repair-test PRIVATE sam3dggml)
                add_test(NAME sam3d_ggml_meshfix_boundary_repair
                         COMMAND $<TARGET_FILE:sam3d-mesh-boundary-repair-test>)
            endif()
        endif()
        if(Python3_Interpreter_FOUND)
            add_test(NAME sam3d_ggml_python_scripts
                     COMMAND ${Python3_EXECUTABLE} -m py_compile
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_regression.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gguf_schema.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/samt_io.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/convert_sam3d_to_gguf.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/plot_benchmarks.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_image_to_3d.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_scene_pipeline.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_scene_benchmark.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/publish_scene_benchmark.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_full_e2e_parity.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_vulkan_cuda_pbr.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/render_glb_compare.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gpu_exclusivity.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/bench_e2e_pipeline.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/bench_full_e2e_pipeline.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/render_compare.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/slat_trajectory_ref.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/slat_step_ref.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/plot_e2e_latency.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate_e2e_benchmark.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/publish_full_e2e.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_full_e2e_report.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_ggml_patch_delivery.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_e2e_matrix.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/materialize_e2e_snapshot.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/profile_e2e.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/ss_trajectory_ref.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/ss_step_ref.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_ss_block_debug.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_slat_block_debug.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gguf_torch_loader.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_operator_oracle.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/q4_ss_trajectory_qat.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate_ss_gguf_trajectory.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate_slat_gguf_trajectory.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_trajectory_validation.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate_gaussian_decoder.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_gaussian_decoder_debug.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gs_decode_ref.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_ss_latent.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/official_reference_policy.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_moge_gguf.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/moge_parity_ref.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/moge_infer_parity.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/moge_block_parity.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_native_preprocess.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/dump_e2e_stages.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/dump_mesh_decoder_reference.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_official_pbr_reference.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_official_pbr_references.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_mesh_decoder_reference.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_mesh_attention_semantics.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_flexicubes_reference.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_native_gaussian_renderer.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_official_pbr_reference.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_mesh_visibility_reference.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_mesh_camera_contract.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_meshfix_reference.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_mesh_export.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_native_pbr_pipeline.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_native_pose_decoder.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_dtype_contract.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_pytorch_cuda_rng.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_pytorch_cuda_randperm.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_pytorch_coordinate_downsample.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_portable_philox_rng.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_flexicubes_tables.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_mesh_ape_reference.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_mesh_visibility_camera_reference.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_native_runtime.py)
            add_test(NAME sam3d_ggml_flexicubes_table_generation
                     COMMAND ${Python3_EXECUTABLE}
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_flexicubes_tables.py
                             --check)
            add_test(NAME sam3d_ggml_trajectory_validation
                     COMMAND ${Python3_EXECUTABLE}
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_trajectory_validation.py)
            add_test(NAME sam3d_ggml_full_e2e_report_validation
                     COMMAND ${Python3_EXECUTABLE}
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_full_e2e_report.py)
            add_test(NAME sam3d_ggml_clean_patch_delivery
                     COMMAND ${Python3_EXECUTABLE}
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_ggml_patch_delivery.py)
            if(SAM3D_GGML_BUILD_CLI
               AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/data/e2e/decode_mesh_vertices.samt"
               AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/data/e2e/decode_mesh_faces.samt"
               AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/data/e2e/decode_mesh_vertex_attrs.samt")
                add_test(NAME sam3d_ggml_native_mesh_asset
                         COMMAND ${Python3_EXECUTABLE}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_mesh_export.py
                                 --binary $<TARGET_FILE:sam3d-cli>
                                 --vertices ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/data/e2e/decode_mesh_vertices.samt
                                 --faces ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/data/e2e/decode_mesh_faces.samt
                                 --attrs ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/data/e2e/decode_mesh_vertex_attrs.samt
                                 --out ${CMAKE_CURRENT_BINARY_DIR}/native_mesh_asset.glb)
            endif()
            if(SAM3D_GGML_CUDA AND SAM3D_GGML_BUILD_CLI)
                add_test(NAME sam3d_native_cuda_no_cudnn
                         COMMAND ${Python3_EXECUTABLE}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_native_runtime.py
                                 --binary $<TARGET_FILE:sam3d-cli>)
                add_test(NAME sam3d_ggml_native_pytorch_cuda_rng
                         COMMAND ${Python3_EXECUTABLE}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_pytorch_cuda_rng.py
                                 --binary $<TARGET_FILE:sam3d-cli>)
                set_tests_properties(sam3d_ggml_native_pytorch_cuda_rng PROPERTIES
                    SKIP_RETURN_CODE 77)
                add_test(NAME sam3d_ggml_native_pytorch_cuda_randperm
                         COMMAND ${Python3_EXECUTABLE}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_pytorch_cuda_randperm.py
                                 --cuda-binary $<TARGET_FILE:sam3d-cli>
                                 --portable-binary $<TARGET_FILE:sam3d-cli>)
                set_tests_properties(sam3d_ggml_native_pytorch_cuda_randperm PROPERTIES
                    SKIP_RETURN_CODE 77)
                add_test(NAME sam3d_ggml_native_pytorch_coordinate_downsample
                         COMMAND ${Python3_EXECUTABLE}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_pytorch_coordinate_downsample.py
                                 --cuda-binary $<TARGET_FILE:sam3d-cli>
                                 --portable-binary $<TARGET_FILE:sam3d-cli>)
                set_tests_properties(sam3d_ggml_native_pytorch_coordinate_downsample PROPERTIES
                    SKIP_RETURN_CODE 77)
                add_test(NAME sam3d_ggml_portable_philox_rng
                         COMMAND ${Python3_EXECUTABLE}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_portable_philox_rng.py
                                 --cuda-binary $<TARGET_FILE:sam3d-cli>
                                 --portable-binary $<TARGET_FILE:sam3d-cli>)
            endif()
        endif()
    endif()
endif()
