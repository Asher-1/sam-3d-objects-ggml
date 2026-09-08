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

# The official textured-GLB path renders source Gaussians with the CUDA
# rasterizer used by SAM 3D Objects.  It is deliberately unavailable in a
# Vulkan-only build: the upstream reference itself is CUDA-only and pretending
# otherwise would hide a backend transfer or a Python fallback.
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
if(SAM3D_GGML_NATIVE_PBR AND NOT SAM3D_GGML_CUDA)
    message(FATAL_ERROR
        "SAM3D_GGML_NATIVE_PBR requires SAM3D_GGML_CUDA=ON: the official "
        "Gaussian renderer and optimized texture bake are CUDA implementations.")
endif()
if(SAM3D_GGML_MESHFIX_GPL AND NOT SAM3D_GGML_NATIVE_PBR)
    message(FATAL_ERROR
        "SAM3D_GGML_MESHFIX_GPL requires SAM3D_GGML_NATIVE_PBR=ON")
endif()
if(SAM3D_GGML_NATIVE_PBR)
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

if(SAM3D_GGML_CUDA)
    set(GGML_CUDA        ON CACHE BOOL "" FORCE)
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "Replay CUDA graphs" FORCE)
endif()
if(SAM3D_GGML_METAL)
    set(GGML_METAL   ON CACHE BOOL "" FORCE)
endif()
if(SAM3D_GGML_VULKAN)
    set(GGML_VULKAN  ON CACHE BOOL "" FORCE)
endif()
if(SAM3D_GGML_HIPBLAS)
    set(GGML_HIP     ON CACHE BOOL "" FORCE)
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

add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

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

if(SAM3D_GGML_NATIVE_PBR)
    find_package(CUDAToolkit REQUIRED)
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
    # These upstream CUDA files are intentionally compiled verbatim. The
    # wrapper in src/gaussian_renderer.cu owns SAM3D-specific file I/O and
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
    src/pointpatch_graph.cpp
    src/ss_flow_graph.cpp
    src/ss_decoder_graph.cpp
    src/sparse_ops.cpp
    src/slat_flow_graph.cpp
    src/gs_decoder_graph.cpp
    src/mesh_decoder_graph.cpp
    src/flexicubes.cpp
    src/mesh_uv.cpp
    src/mesh_boundary_repair.cpp
    src/session.cpp
)

if(SAM3D_GGML_NATIVE_PBR)
    list(APPEND SAM3DGGML_SOURCES
        src/mesh_postprocess.cpp
        src/texture_inpaint.cpp
        src/gaussian_renderer.cpp
        src/gaussian_renderer.cu)
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
        add_executable(sam3d-asset-io-test src/asset_io_test.cpp)
        target_link_libraries(sam3d-asset-io-test PRIVATE sam3dggml)
        add_test(NAME sam3d_ggml_native_glb_asset
                 COMMAND $<TARGET_FILE:sam3d-asset-io-test>)
        add_executable(sam3d-image-preprocess-test src/image_preprocess_test.cpp)
        target_link_libraries(sam3d-image-preprocess-test PRIVATE sam3dggml)
        add_test(NAME sam3d_ggml_native_image_preprocess
                 COMMAND $<TARGET_FILE:sam3d-image-preprocess-test>)
        add_executable(sam3d-mesh-tables-test src/mesh_tables_test.cpp)
        target_link_libraries(sam3d-mesh-tables-test PRIVATE sam3dggml)
        add_test(NAME sam3d_ggml_mesh_subdivide_tables
                 COMMAND $<TARGET_FILE:sam3d-mesh-tables-test>)
        add_executable(sam3d-flexicubes-test src/flexicubes_test.cpp)
        target_link_libraries(sam3d-flexicubes-test PRIVATE sam3dggml)
        add_test(NAME sam3d_ggml_flexicubes_synthetic_mesh
                 COMMAND $<TARGET_FILE:sam3d-flexicubes-test>)
        add_executable(sam3d-mesh-uv-test src/mesh_uv_test.cpp)
        target_link_libraries(sam3d-mesh-uv-test PRIVATE sam3dggml)
        add_test(NAME sam3d_ggml_xatlas_uv
                 COMMAND $<TARGET_FILE:sam3d-mesh-uv-test>)
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
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/convert_sam3d_to_gguf.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/plot_benchmarks.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_image_to_3d.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gpu_exclusivity.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/bench_e2e_pipeline.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/render_compare.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/slat_trajectory_ref.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/plot_e2e_latency.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate_e2e_benchmark.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_e2e_matrix.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/profile_e2e.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/q4_ss_trajectory_qat.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate_ss_gguf_trajectory.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/moge_parity_ref.py
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
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_meshfix_reference.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_mesh_export.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_flexicubes_tables.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_native_runtime.py)
            add_test(NAME sam3d_ggml_flexicubes_table_generation
                     COMMAND ${Python3_EXECUTABLE}
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_flexicubes_tables.py
                             --check)
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
            endif()
        endif()
    endif()
endif()
