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

set(_sam3d_gpu_backends 0)
foreach(_backend SAM3D_GGML_CUDA SAM3D_GGML_METAL SAM3D_GGML_VULKAN SAM3D_GGML_HIPBLAS)
    if(${_backend})
        math(EXPR _sam3d_gpu_backends "${_sam3d_gpu_backends} + 1")
    endif()
endforeach()
if(_sam3d_gpu_backends GREATER 1)
    message(FATAL_ERROR "Select at most one GPU backend per build directory")
endif()

# ggml uses CMake's `native` CUDA architecture only with CMake >=3.24. On older
# CMake, select the attached GPU instead of compiling a broad arch list.
if(SAM3D_GGML_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_LESS 3.24)
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

set(SAM3DGGML_SOURCES
    src/common.cpp
    src/asset_io.cpp
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
    src/session.cpp
)

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
    PRIVATE sam3d_stb)

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
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_mesh_export.py
                             ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_native_runtime.py)
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
