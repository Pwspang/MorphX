#pragma once
#include <stdlib.h>
#include <unistd.h>

/**
 * Path of the real CUDA driver. Shared by the hook driver (preload.h) and the
 * dlopen shim (preload.c): the shim only lets a dlopen through unhooked when the
 * path matches exactly, so both sides must resolve it the same way.
 * NEUTRINO_REAL_DRIVER overrides; otherwise the first existing Debian/Ubuntu or
 * RHEL/Rocky location is used.
 */
static const char* morphx_real_driver_path(void) {
    static const char* const candidates[] = {
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so",
        "/usr/lib64/libcuda.so.1",
    };
    const char* env = getenv("NEUTRINO_REAL_DRIVER");
    if (env != NULL && env[0] != '\0') {
        return env;
    }
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        if (access(candidates[i], F_OK) == 0) {
            return candidates[i];
        }
    }
    return candidates[0];
}
