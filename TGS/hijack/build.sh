#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

# The hijack takes the libcuda.so.1 soname, so libraries that dlopen("libcuda.so.1")
# and dlsym() driver entry points by name get the hijack. It only implements a
# pre-CUDA-12 subset, so record the real driver as a DT_NEEDED dependency (by
# absolute path, since its soname is libcuda.so.1 too): dlsym on the hijack's
# handle then falls back to the real driver for everything the hijack doesn't
# export (cuBLAS 13 needs cuTensorMapEncodeTiled, cuKernelSetAttribute, ...).
# TGS_REAL_LIBCUDA overrides the auto-detected path.
REAL_LIBCUDA=${TGS_REAL_LIBCUDA:-}
if [[ -z "${REAL_LIBCUDA}" ]]; then
    for candidate in /usr/lib64/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1; do
        if [[ -e "${candidate}" ]]; then
            REAL_LIBCUDA=${candidate}
            break
        fi
    done
fi
if [[ -z "${REAL_LIBCUDA}" ]]; then
    echo "build.sh: real libcuda.so.1 not found; set TGS_REAL_LIBCUDA" >&2
    exit 1
fi

# TGS_HIJACK_NVML=1 also builds and installs the NVML wrappers (see CMakeLists.txt).
HIJACK_NVML=${TGS_HIJACK_NVML:-0}

function install_libs() {
    local ROOT=$1 PRIORITY=$2

    rm -rf ${ROOT}/${PRIORITY}-priority-lib
    mkdir ${ROOT}/${PRIORITY}-priority-lib
    cd ${ROOT}/${PRIORITY}-priority-lib
    cp ${ROOT}/build/libcuda-control-${PRIORITY}-priority.so ./libcuda-control.so
    patchelf --add-needed "${REAL_LIBCUDA}" ./libcuda-control.so

    touch ./ld.so.preload
    if [[ "${HIJACK_NVML}" == 1 ]]; then
        echo -e "/libcontroller.so\n/libcuda.so\n/libcuda.so.1\n/libnvidia-ml.so\n/libnvidia-ml.so.1" > ./ld.so.preload
        cp libcuda-control.so ./libnvidia-ml.so.1
        patchelf --set-soname libnvidia-ml.so.1 ./libnvidia-ml.so.1
        cp libcuda-control.so ./libnvidia-ml.so
        patchelf --set-soname libnvidia-ml.so ./libnvidia-ml.so
    else
        echo -e "/libcontroller.so\n/libcuda.so\n/libcuda.so.1" > ./ld.so.preload
    fi
    cp libcuda-control.so ./libcuda.so.1
    patchelf --set-soname libcuda.so.1 ./libcuda.so.1
    cp libcuda-control.so ./libcuda.so
    patchelf --set-soname libcuda.so ./libcuda.so
    cp libcuda-control.so ./libcontroller.so
    patchelf --set-soname libcontroller.so ./libcontroller.so
}

function build() {
    ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)
    rm -rf ${ROOT}/build
    mkdir ${ROOT}/build
    cd ${ROOT}/build
    if [[ "${HIJACK_NVML}" == 1 ]]; then
        cmake -DCMAKE_BUILD_TYPE=Release -DTGS_HIJACK_NVML=ON ..
    else
        cmake -DCMAKE_BUILD_TYPE=Release ..
    fi
    make

    install_libs ${ROOT} high
    install_libs ${ROOT} low

    cd ..
    # BASE_IMAGE=${BASE_IMAGE:-"tf_torch"}
    # docker build -t ${BASE_IMAGE} --network=host -f ./Dockerfile .
}

build
