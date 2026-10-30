cmake_minimum_required(VERSION 3.20)

option(SMSCHED_ENABLE_PASS "Build CUDA kernels with the SMSched LLVM pass" OFF)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/morphx.cmake)

if(SMSCHED_ENABLE_PASS)
    # Pass builds need clang; still overridable with -DCMAKE_<LANG>_COMPILER.
    set(SMSCHED_LLVM_ROOT "${MORPHX_LLVM_ROOT}" CACHE PATH "LLVM install prefix for SMSched pass builds")
    set(CMAKE_CUDA_COMPILER "${SMSCHED_LLVM_ROOT}/bin/clang++" CACHE FILEPATH "")
    set(CMAKE_CXX_COMPILER "${SMSCHED_LLVM_ROOT}/bin/clang++" CACHE FILEPATH "")
    set(CMAKE_C_COMPILER "${SMSCHED_LLVM_ROOT}/bin/clang" CACHE FILEPATH "")
elseif(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX} AND EXISTS "${MORPHX_CUDA_ROOT}/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "${MORPHX_CUDA_ROOT}/bin/nvcc" CACHE FILEPATH "")
endif()
set(CMAKE_CUDA_ARCHITECTURES "${MORPHX_CUDA_ARCH}" CACHE STRING "CUDA SM architectures to target")
set(CUDAToolkit_ROOT "${MORPHX_CUDA_ROOT}" CACHE PATH "CUDA Toolkit root directory")

project(microbench LANGUAGES C CXX CUDA)

if(SMSCHED_ENABLE_PASS)
    get_filename_component(SMSCHED_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
    set(SMSCHED_PASS_SO "${SMSCHED_REPO_ROOT}/smsched-pass/build/libsmsched.so" CACHE FILEPATH "SMSched LLVM pass plugin")
    set(SMSCHED_CUDA_COMMON_FLAGS
        ${MORPHX_STDLIB_COMPILE_FLAGS}
        -I${MORPHX_CUDA_ROOT}/include
        -I${MORPHX_CUDA_ROOT}/targets/x86_64-linux/include
        ${MORPHX_CCCL_INCLUDE_FLAGS}
        -std=c++17
        -O3
        -fplugin=${SMSCHED_PASS_SO}
        -fpass-plugin=${SMSCHED_PASS_SO}
        -mllvm
        -enable-dump-ir
        -mllvm
        -dump-ir-dir=${CMAKE_CURRENT_BINARY_DIR}/ir
        ${MORPHX_CLANG_CUDA_FLAGS}
        -include
        cuda_fp16.h
    )
    string(REPLACE ";" " " SMSCHED_CUDA_FLAGS_STR "${SMSCHED_CUDA_COMMON_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${SMSCHED_CUDA_FLAGS_STR}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MORPHX_STDLIB_COMPILE_FLAGS} ${MORPHX_STDLIB_CXX_FLAGS} -I${MORPHX_CUDA_ROOT}/include -include cuda_fp16.h")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${MORPHX_CUDA_ROOT}/lib64 ${MORPHX_STDLIB_LINK_FLAGS} -lcudart -lcuda ${MORPHX_GDR_LINK}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${MORPHX_CUDA_ROOT}/lib64 ${MORPHX_STDLIB_LINK_FLAGS} -lcudart -lcuda ${MORPHX_GDR_LINK}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CUTLASS_ROOT "${MORPHX_CUTLASS_ROOT}" CACHE PATH "CUTLASS install prefix")
set(CUTLASS_INCLUDE_DIRS
    "${CUTLASS_ROOT}/include"
    "${CUTLASS_ROOT}/tools/util/include"
)

find_package(CUDAToolkit REQUIRED cublas)

# sm80 (default): CUTLASS 2 Sm80 kernels, what Figure 16 measures. sm90: CUTLASS 3
# WGMMA/TMA kernels (src/my_cutlass_gemm_sm90.cu), same C API; needs the clang +
# pass build and MORPHX_CUDA_ARCH=90a, into its own build dir, e.g.
#   cmake -S microbench -B microbench/build-model-sm90 -DSMSCHED_ENABLE_PASS=ON \
#         -DMORPHX_MYCUTLASS_ARCH=sm90 -DMORPHX_CUDA_ARCH=90a
set(MORPHX_MYCUTLASS_ARCH "sm80" CACHE STRING "libmycutlass kernels: sm80 (CUTLASS 2) or sm90 (CUTLASS 3 WGMMA/TMA)")
set_property(CACHE MORPHX_MYCUTLASS_ARCH PROPERTY STRINGS sm80 sm90)
if(MORPHX_MYCUTLASS_ARCH STREQUAL "sm90")
    if(NOT SMSCHED_ENABLE_PASS OR NOT MORPHX_CUDA_ARCH STREQUAL "90a")
        message(FATAL_ERROR "MORPHX_MYCUTLASS_ARCH=sm90 needs -DSMSCHED_ENABLE_PASS=ON -DMORPHX_CUDA_ARCH=90a")
    endif()
    add_library(mycutlass SHARED src/my_cutlass_gemm_sm90.cu)
    target_compile_definitions(mycutlass PRIVATE
        # device asserts make ptxas serialize the wgmma pipeline (C7510)
        NDEBUG
        CUTLASS_GRID_CONSTANT=__grid_constant__
        # call cuTensorMapEncodeTiled directly instead of looking it up with
        # cudaGetDriverEntryPoint for every TMA descriptor of every GEMM call
        CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1
        # CUTLASS sees CUDA 12.4 (see cmake/morphx.cmake); CUDA 13 keeps the _v12000 typedefs
        PFN_cuTensorMapEncodeTiled=PFN_cuTensorMapEncodeTiled_v12000
        PFN_cuTensorMapEncodeIm2col=PFN_cuTensorMapEncodeIm2col_v12000)
    # CUTLASS 3 kernels keep tiles/mbarriers in dynamic smem word 0, which the pass
    # also uses for the block-id hand-off (see SmemHandoffBarriers in SMSched.cpp)
    target_compile_options(mycutlass PRIVATE "SHELL:-mllvm -smsched-smem-handoff-barriers")
    target_link_libraries(mycutlass PUBLIC CUDA::cuda_driver)
elseif(MORPHX_MYCUTLASS_ARCH STREQUAL "sm80")
    add_library(mycutlass SHARED src/my_cutlass_gemm.cu)
else()
    message(FATAL_ERROR "MORPHX_MYCUTLASS_ARCH must be sm80 or sm90")
endif()
# Shared library provides the kernels consumed by the executables.
target_include_directories(mycutlass
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUTLASS_INCLUDE_DIRS}
)
target_link_libraries(mycutlass
    PUBLIC
        CUDA::cudart
)

add_executable(main src/main.cu)
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(main PRIVATE mycutlass)

add_executable(bench src/bench.cu)
target_include_directories(bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bench PRIVATE mycutlass CUDA::cublas)

add_executable(stream_bench src/stream_bench.cu)
target_include_directories(stream_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(stream_bench PRIVATE mycutlass CUDA::cublas)

add_executable(stream_mm_test src/stream_mm_test.cu)
target_include_directories(stream_mm_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(stream_mm_test PRIVATE mycutlass)

add_executable(green_ctx_bench src/green_ctx_bench.cu)
target_include_directories(green_ctx_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(green_ctx_bench PRIVATE mycutlass CUDA::cuda_driver)

install(TARGETS mycutlass main bench stream_mm_test green_ctx_bench
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)
