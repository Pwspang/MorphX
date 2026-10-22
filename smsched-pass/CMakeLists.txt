cmake_minimum_required(VERSION 3.10)
project(smsched-pass)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# LLVM_DIR wins; otherwise use MORPHX_LLVM_ROOT (env or cache), like the
# other MorphX builds.
if(DEFINED ENV{MORPHX_LLVM_ROOT})
    set(MORPHX_LLVM_ROOT_DEFAULT "$ENV{MORPHX_LLVM_ROOT}")
else()
    set(MORPHX_LLVM_ROOT_DEFAULT "/home/rtx/llvm-project/install")
endif()
set(MORPHX_LLVM_ROOT "${MORPHX_LLVM_ROOT_DEFAULT}" CACHE PATH "LLVM install prefix used with the MorphX pass")
find_package(LLVM REQUIRED CONFIG HINTS "${MORPHX_LLVM_ROOT}/lib/cmake/llvm")
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
if(NOT ${LLVM_ENABLE_RTTI})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

add_llvm_pass_plugin(libsmsched SMSched.cpp)