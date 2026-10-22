cmake_minimum_required(VERSION 3.5)
project(hijack C CXX)

add_definitions(-D_GNU_SOURCE)

# Directory for the TGS rate-watcher Unix sockets; $MORPHX_TGS_SOCKET_DIR at
# run time overrides it.
if(DEFINED ENV{MORPHX_TGS_SOCKET_DIR})
    set(MORPHX_TGS_SOCKET_DIR_DEFAULT "$ENV{MORPHX_TGS_SOCKET_DIR}")
else()
    set(MORPHX_TGS_SOCKET_DIR_DEFAULT "/home/rtx/gpu/sm-sched/TGS/gsharing")
endif()
set(MORPHX_TGS_SOCKET_DIR "${MORPHX_TGS_SOCKET_DIR_DEFAULT}" CACHE PATH "Directory for TGS rate-watcher sockets")
add_definitions(-DMORPHX_TGS_SOCKET_DIR="${MORPHX_TGS_SOCKET_DIR}")
add_compile_options(-Wall -Wshadow -Werror -Wno-format)

# The NVML wrappers never had an entry table (nvml_library_entry is empty, and
# loading it called dlsym with NULL names), so every exported nvml* symbol jumps
# through NULL: preloading the libnvidia-ml copies alone segfaults torch on the
# H200. TGS itself doesn't use NVML, so leave it out unless asked for
# (build.sh: TGS_HIJACK_NVML=1).
option(TGS_HIJACK_NVML "Export the (unpopulated) NVML wrappers" OFF)
if(TGS_HIJACK_NVML)
    add_definitions(-DTGS_HIJACK_NVML)
    set(TGS_NVML_SOURCES src/nvml_entry.c)
else()
    set(TGS_NVML_SOURCES "")
endif()

include_directories(${CMAKE_SOURCE_DIR})

if (${ENABLE_DEBUG})
    add_compile_options(-g -O0)
else ()
    add_compile_options(-g -O2)
endif ()

set(STATIC_C_LIBRARIES -static-libgcc -static-libstdc++)

add_library(cuda-control-high-priority SHARED
        src/high_priority_hijack_call.c
        include/hijack.h
        include/cuda-subset.h
        include/nvml-subset.h
        include/cuda-helper.h
        include/nvml-helper.h
        src/cuda_originals.c
        ${TGS_NVML_SOURCES}
        src/loader.c
        src/proc_address.c)

target_link_libraries(cuda-control-high-priority ${STATIC_C_LIBRARIES})

target_compile_options(cuda-control-high-priority PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>)

add_library(cuda-control-low-priority SHARED
        src/low_priority_hijack_call.c
        include/hijack.h
        include/cuda-subset.h
        include/nvml-subset.h
        include/cuda-helper.h
        include/nvml-helper.h
        src/cuda_originals.c
        ${TGS_NVML_SOURCES}
        src/loader.c
        src/proc_address.c)

target_link_libraries(cuda-control-low-priority ${STATIC_C_LIBRARIES})

target_compile_options(cuda-control-low-priority PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>)

