cmake_minimum_required(VERSION 3.22)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/morphx.cmake)

# Prefer gcc-12 when present (original evaluation machine); CC/CXX or
# -DCMAKE_<LANG>_COMPILER override it, otherwise the default compiler is used.
if(NOT DEFINED CMAKE_C_COMPILER AND NOT DEFINED ENV{CC} AND EXISTS "/usr/bin/gcc-12")
    set(CMAKE_C_COMPILER "/usr/bin/gcc-12")
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX} AND EXISTS "/usr/bin/g++-12")
    set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")
endif()

project(smsched-runtime)

set(CUDAToolkit_ROOT "${MORPHX_CUDA_ROOT}" CACHE PATH "CUDA Toolkit root directory")
find_package(CUDAToolkit 12.2 REQUIRED)

# The fake libcuda.so links against the real driver library. Prefer the
# installed driver, fall back to the toolkit stub (same SONAME libcuda.so.1).
find_library(MORPHX_REAL_LIBCUDA
    NAMES cuda
    HINTS /usr/lib/x86_64-linux-gnu /usr/lib64
    NO_DEFAULT_PATH
)
if(NOT MORPHX_REAL_LIBCUDA)
    find_library(MORPHX_REAL_LIBCUDA NAMES cuda HINTS ${CUDAToolkit_LIBRARY_DIR}/stubs)
endif()
if(NOT MORPHX_REAL_LIBCUDA)
    message(FATAL_ERROR "libcuda.so not found; set MORPHX_REAL_LIBCUDA")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 20)
set(PROJ_NAME "smsched-runtime")

set(SOURCE src/stub.cpp)
add_library(stub SHARED)
target_sources(stub PRIVATE ${SOURCE})
target_include_directories(stub PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ../include
    ../3rdparty/
)
target_link_directories(stub PRIVATE
    ${CUDAToolkit_LIBRARY_DIR}
)
target_link_libraries(stub
    PRIVATE
    dl
    cudart
)
set_property(TARGET stub PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(stub_raw SHARED)
target_sources(stub_raw PRIVATE src/stub_raw.cpp)
target_include_directories(stub_raw PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ../include
    ../3rdparty/
)
target_link_directories(stub_raw PRIVATE
    ${CUDAToolkit_LIBRARY_DIR}
)
target_link_libraries(stub_raw
    PRIVATE
    dl
    cudart
)
set_property(TARGET stub_raw PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(api SHARED)
target_sources(api PRIVATE
    src/api.cpp
)
target_include_directories(api PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ../include
)
target_link_directories(api PRIVATE
    ${CUDAToolkit_LIBRARY_DIR}
)
target_link_libraries(api
    PRIVATE
    dl
    cudart
)
set_property(TARGET api PROPERTY POSITION_INDEPENDENT_CODE ON)


add_library(base SHARED)
target_sources(base PRIVATE src/base.cpp)
target_include_directories(base PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ../include
)
target_link_directories(base PRIVATE
    ${CUDAToolkit_LIBRARY_DIR}
)
target_link_libraries(base
    PRIVATE
    dl
    cudart
)
set_property(TARGET base PROPERTY POSITION_INDEPENDENT_CODE ON)

# ========== Elegant CMake targets for libcuda.so and libpreload.so ==========

# Option to use cuda-llm.cpp instead of cuda.cpp
option(USE_CUDA_LOG "Use cuda-log.cpp instead of cuda.cpp for compilation" OFF)
option(USE_CUDA_LLM "Use cuda-llm.cpp instead of cuda.cpp for compilation" OFF)
option(USE_CUDA_ORION "Use cuda-orion.cpp instead of cuda.cpp for compilation" OFF)
option(USE_CUDA_LITHOS "Use cuda-lithos.cpp instead of cuda.cpp for compilation" OFF)

# libcuda.so (C++)
if(USE_CUDA_LOG)
    set(CUDA_SOURCE src/cuda-log.cpp)
elseif(USE_CUDA_LITHOS)
    set(CUDA_SOURCE src/cuda-lithos.cpp)
elseif(USE_CUDA_LLM)
    set(CUDA_SOURCE src/cuda-llm.cpp)
elseif(USE_CUDA_ORION)
    set(CUDA_SOURCE src/cuda-orion.cpp)
else()
    set(CUDA_SOURCE src/cuda.cpp)
endif()
add_library(cuda SHARED ${CUDA_SOURCE})
set_target_properties(cuda PROPERTIES OUTPUT_NAME "cuda" PREFIX "lib")
target_include_directories(cuda PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ../include)
target_link_directories(cuda PRIVATE ${CUDAToolkit_LIBRARY_DIR})
target_compile_options(cuda PRIVATE -O3 -g -fPIC)
target_link_libraries(cuda PRIVATE cudart dl stdc++ ${MORPHX_REAL_LIBCUDA})

# Narrow LD_PRELOAD shim for workloads where preloading the full fake libcuda
# changes too many driver symbols. It reuses the scheduler implementation but
# only exports the interception points needed by cudart/cuGetProcAddress.
add_library(cuda_hook SHARED ${CUDA_SOURCE})
set_target_properties(cuda_hook PROPERTIES OUTPUT_NAME "cuda_hook" PREFIX "lib" NO_SONAME TRUE)
target_include_directories(cuda_hook PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ../include)
target_link_directories(cuda_hook PRIVATE ${CUDAToolkit_LIBRARY_DIR})
target_compile_options(cuda_hook PRIVATE -O3 -g -fPIC)
target_link_libraries(cuda_hook PRIVATE cudart dl stdc++ ${MORPHX_REAL_LIBCUDA})
target_link_options(cuda_hook PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_hook_exports.map"
    "-Wl,-soname,libcuda.so"
)

# libpreload.so (C)
add_library(preload SHARED src/preload.c)
set_target_properties(preload PROPERTIES OUTPUT_NAME "preload" PREFIX "lib")
target_compile_options(preload PRIVATE -O3 -g -fPIC)
