# Machine-specific build settings shared by runtime/, microbench/ and ggnn/.
# Each MORPHX_* value is a cache variable whose default comes from the
# environment variable of the same name, falling back to the original A100
# evaluation machine. Safe to include before project().

include_guard(GLOBAL)

# morphx_option(<var> <A100 default> <type> <doc>)
macro(morphx_option var default type doc)
    if(DEFINED ENV{${var}})
        set(_morphx_default "$ENV{${var}}")
    else()
        set(_morphx_default "${default}")
    endif()
    set(${var} "${_morphx_default}" CACHE ${type} "${doc}")
endmacro()

morphx_option(MORPHX_LLVM_ROOT "/home/rtx/llvm-project/install" PATH "LLVM install prefix (clang + libc++) used with the MorphX pass")
morphx_option(MORPHX_CUDA_ROOT "/usr/local/cuda-12.4" PATH "CUDA toolkit root")
morphx_option(MORPHX_CUTLASS_ROOT "/home/rtx/gpu/cutlass" PATH "CUTLASS source/install prefix")
morphx_option(MORPHX_CUDA_ARCH "80" STRING "CUDA SM architecture to target, e.g. 80 (A100) or 90 (H100/H200)")

# CUDA toolkit version from cuda.h, so it is known before project().
set(MORPHX_CUDA_VERSION_MAJOR 0)
set(MORPHX_CUDA_VERSION_MINOR 0)
if(EXISTS "${MORPHX_CUDA_ROOT}/include/cuda.h")
    file(STRINGS "${MORPHX_CUDA_ROOT}/include/cuda.h" _morphx_cuda_version_line
        REGEX "^#define CUDA_VERSION [0-9]+")
    string(REGEX REPLACE "^#define CUDA_VERSION ([0-9]+).*" "\\1" _morphx_cuda_version "${_morphx_cuda_version_line}")
    math(EXPR MORPHX_CUDA_VERSION_MAJOR "${_morphx_cuda_version} / 1000")
    math(EXPR MORPHX_CUDA_VERSION_MINOR "(${_morphx_cuda_version} % 1000) / 10")
else()
    message(WARNING "MorphX: ${MORPHX_CUDA_ROOT}/include/cuda.h not found; set MORPHX_CUDA_ROOT")
endif()

# CCCL (Thrust/CUB/libcudacxx) used by the clang + pass builds. CUDA 13 ships
# CCCL 3.x under include/cccl; older toolkits used a separate CCCL checkout.
if(DEFINED ENV{MORPHX_CCCL_ROOT})
    set(_morphx_cccl_default "$ENV{MORPHX_CCCL_ROOT}")
elseif(MORPHX_CUDA_VERSION_MAJOR GREATER_EQUAL 13)
    set(_morphx_cccl_default "")
else()
    set(_morphx_cccl_default "/home/rtx/gpu/cccl")
endif()
set(MORPHX_CCCL_ROOT "${_morphx_cccl_default}" CACHE PATH "CCCL checkout; empty uses the CUDA toolkit's bundled CCCL")
if(MORPHX_CCCL_ROOT AND NOT EXISTS "${MORPHX_CCCL_ROOT}/thrust/thrust")
    # Installed/bundled layout (thrust/, cub/, cuda/ side by side).
    set(MORPHX_CCCL_INCLUDE_FLAGS -I${MORPHX_CCCL_ROOT})
elseif(MORPHX_CCCL_ROOT)
    # Source checkout layout.
    set(MORPHX_CCCL_INCLUDE_FLAGS
        -I${MORPHX_CCCL_ROOT}/thrust
        -I${MORPHX_CCCL_ROOT}/libcudacxx/include
        -I${MORPHX_CCCL_ROOT}/cub
    )
elseif(EXISTS "${MORPHX_CUDA_ROOT}/include/cccl")
    set(MORPHX_CCCL_INCLUDE_FLAGS -I${MORPHX_CUDA_ROOT}/include/cccl)
else()
    set(MORPHX_CCCL_INCLUDE_FLAGS "")
endif()

# Scheduler control-word backend: gdr (GDRCopy, default) or hostmapped (pinned
# host memory, for machines without the gdrdrv kernel module).
morphx_option(MORPHX_SYNC_BACKEND "gdr" STRING "Scheduler control-word backend: gdr or hostmapped")
set_property(CACHE MORPHX_SYNC_BACKEND PROPERTY STRINGS gdr hostmapped)
if(MORPHX_SYNC_BACKEND STREQUAL "hostmapped")
    add_compile_definitions(MORPHX_SYNC_BACKEND_HOSTMAPPED)
    set(MORPHX_GDR_LINK "")
elseif(MORPHX_SYNC_BACKEND STREQUAL "gdr")
    set(MORPHX_GDR_LINK "-lgdrapi")
else()
    message(FATAL_ERROR "Unknown MORPHX_SYNC_BACKEND: ${MORPHX_SYNC_BACKEND}")
endif()

# The original (CUDA 12) pass builds compile against LLVM's libc++ and pretend
# to be nvcc (-D__NVCC__ ...). CUDA 13's CCCL headers break under that
# pretence, while clang's own CUDA mode works with libstdc++, so CUDA >= 13
# defaults to the plain mode.
if(MORPHX_CUDA_VERSION_MAJOR GREATER_EQUAL 13)
    set(_morphx_libcxx_default OFF)
else()
    set(_morphx_libcxx_default ON)
endif()
morphx_option(MORPHX_CLANG_LIBCXX "${_morphx_libcxx_default}" BOOL "Build pass-enabled code with libc++ and the nvcc-compat defines (original CUDA 12 setup)")

# Flags shared by every clang CUDA compile that runs the MorphX pass, plus the
# standard-library flags for compiles (MORPHX_STDLIB_COMPILE_FLAGS, CUDA and C++),
# C++-only extras (MORPHX_STDLIB_CXX_FLAGS) and links (MORPHX_STDLIB_LINK_FLAGS).
if(MORPHX_CLANG_LIBCXX)
    set(MORPHX_CLANG_CUDA_FLAGS
        --cuda-path=${MORPHX_CUDA_ROOT}
        --cuda-gpu-arch=sm_${MORPHX_CUDA_ARCH}
        -D__NVCC__
        -D__CUDACC_VER_MAJOR__=${MORPHX_CUDA_VERSION_MAJOR}
        -D__CUDACC_VER_MINOR__=${MORPHX_CUDA_VERSION_MINOR}
        -D__NV_NO_HOST_COMPILER_CHECK
        -D_ALLOW_UNSUPPORTED_LIBCPP
        -D_VSTD=std
        -Wno-invalid-specialization
        -D__CUDACC__
    )
    set(MORPHX_STDLIB_COMPILE_FLAGS "-nostdinc++ -I${MORPHX_LLVM_ROOT}/include/c++/v1 -I${MORPHX_LLVM_ROOT}/include/x86_64-unknown-linux-gnu/c++/v1")
    set(MORPHX_STDLIB_CXX_FLAGS "-D_VSTD=std -Wno-error=missing-template-arg-list-after-template-kw -Wno-invalid-specialization")
    set(MORPHX_STDLIB_LINK_FLAGS "-stdlib=libc++ -lc++ -lc++abi")
else()
    set(MORPHX_CLANG_CUDA_FLAGS
        --cuda-path=${MORPHX_CUDA_ROOT}
        --cuda-gpu-arch=sm_${MORPHX_CUDA_ARCH}
        -Wno-unknown-cuda-version
    )
    set(MORPHX_STDLIB_COMPILE_FLAGS "")
    set(MORPHX_STDLIB_CXX_FLAGS "")
    set(MORPHX_STDLIB_LINK_FLAGS "")
endif()

message(STATUS "MorphX: CUDA ${MORPHX_CUDA_VERSION_MAJOR}.${MORPHX_CUDA_VERSION_MINOR} at ${MORPHX_CUDA_ROOT}, sm_${MORPHX_CUDA_ARCH}, sync backend ${MORPHX_SYNC_BACKEND}")
