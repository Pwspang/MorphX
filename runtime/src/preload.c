#define _GNU_SOURCE
#include <dlfcn.h>     // for dynamic library
#include <stdio.h>     // for I/O
#include <string.h>    // for strcmp
#include <execinfo.h>  // for backtrace and backtrace_symbols
#include <stdlib.h>    // for malloc and free
#include <time.h>      // for timing terms
#include <assert.h>
#include <limits.h>

#define CHECK_DL() do {                    \
    const char *dl_error = dlerror();      \
    if (dl_error) {                        \
        fprintf(stderr, "%s\n", dl_error); \
        exit(EXIT_FAILURE);                \
    }                                      \
} while (0)

#ifndef STACK_TRACE_SIZE
#define STACK_TRACE_SIZE 5
#endif

#ifndef DL_VERBOSE
#define DL_VERBOSE 0
#endif

// Pointer to GLIBC dlopen function, by dlsym(RTLD_NEXT, "dlopen")
static void* (*real_dlopen)(const char *filename, int flags) = NULL;

static const char* NEUTRINO_REAL_DRIVER = "/usr/lib/x86_64-linux-gnu/libcuda.so";
static const char* NEUTRINO_HOOK_DRIVER = NULL;
static const char* NEUTRINO_DRIVER_NAME = "libcuda.so.1";
static char hook_driver_path[PATH_MAX];

// The hook driver is built next to this preload shim. Derive that location so
// a fresh checkout never falls back to a library from another working tree.
__attribute__((constructor))
static void initialize_hook_driver_path(void) {
    Dl_info info;
    if (dladdr((void*)&initialize_hook_driver_path, &info) == 0 || info.dli_fname == NULL) {
        return;
    }

    const char* separator = strrchr(info.dli_fname, '/');
    if (separator == NULL) {
        return;
    }

    static const char hook_name[] = "libcuda.so";
    size_t directory_length = (size_t)(separator - info.dli_fname) + 1;
    if (directory_length + sizeof(hook_name) > sizeof(hook_driver_path)) {
        return;
    }

    memcpy(hook_driver_path, info.dli_fname, directory_length);
    memcpy(hook_driver_path + directory_length, hook_name, sizeof(hook_name));
    NEUTRINO_HOOK_DRIVER = hook_driver_path;
}

void* dlopen(const char *filename, int flags) {
    if (!real_dlopen) 
        real_dlopen = dlsym(RTLD_NEXT, "dlopen");

    const char* hook_driver_env = getenv("NEUTRINO_HOOK_DRIVER");
    const char* hook_driver =
        hook_driver_env != NULL && hook_driver_env[0] != '\0' ? hook_driver_env : NEUTRINO_HOOK_DRIVER;
    const char* real_driver_env = getenv("NEUTRINO_REAL_DRIVER");
    const char* real_driver =
        real_driver_env != NULL && real_driver_env[0] != '\0' ? real_driver_env : NEUTRINO_REAL_DRIVER;
    const char* driver_name_env = getenv("NEUTRINO_DRIVER_NAME");
    const char* driver_name =
        driver_name_env != NULL && driver_name_env[0] != '\0' ? driver_name_env : NEUTRINO_DRIVER_NAME;

    // if (filename != NULL && (strstr(filename, "libcuda.so") != NULL)) {
    //     printf("[dlopen] Intercepted loading of %s\n", filename);
    //     fflush(stdout);
    // }

    int is_real_driver_path = (
        filename != NULL
        && real_driver != NULL
        && strcmp(filename, real_driver) == 0
    );
    if (filename != NULL && !is_real_driver_path && (strstr(filename, driver_name) != NULL || strstr(filename, "libcuda.so") != NULL)) {
        
        // Keep AE logs focused on workload output; enable DL_VERBOSE for loader diagnostics.

        void* array[STACK_TRACE_SIZE];
        int size       = backtrace(array, STACK_TRACE_SIZE);
        char** strings = backtrace_symbols(array, size);
        int call_from_cublas = 0;
        if (strings != NULL){
            for (int i = 0; i < size; i++) {
                // we will add ALL Nvidia Propietray Product here
                if (strstr(strings[i], "libcublas") != NULL) {
                    call_from_cublas = 1;
                    break;
                }
            }
        }
        free(strings);
        char* force_hook_cublas = getenv("SMSCHED_FORCE_HOOK_CUBLAS");
        if (force_hook_cublas != NULL && atoi(force_hook_cublas) != 0) {
            call_from_cublas = 0;
        }
        void* ptr;
        if (call_from_cublas) {
            if (real_driver == NULL) {
                fprintf(stderr, "[error] NEUTRINO_REAL_DRIVER not set\n");
                exit(1);
            }
            ptr = real_dlopen(real_driver, flags);
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            long long time = ts.tv_nsec + ts.tv_sec * 1e9;
            // printf("[info] %lld cublas use real: %s %p %d\n", time, NEUTRINO_REAL_DRIVER, ptr, flags);
            fflush(stdout);
        } else {
            if (hook_driver == NULL) {
                fprintf(stderr, "[error] unable to locate the MorphX hook driver; set NEUTRINO_HOOK_DRIVER\n");
                return real_dlopen(filename, flags);
            }
            // @note fix the multiple initialization bug
            ptr = real_dlopen(hook_driver, flags | RTLD_GLOBAL);
            CHECK_DL();

            // fprintf(stderr, "[dlopen] %s : %d, %p\n", NEUTRINO_HOOK_DRIVER, flags | RTLD_GLOBAL, ptr);
            if (DL_VERBOSE) {
                struct timespec ts;
                clock_gettime(CLOCK_REALTIME, &ts);
                long long time = ts.tv_nsec + ts.tv_sec * 1e9;
                printf("[info] %lld use hooked: %s %p %d\n", time, hook_driver, ptr, flags);
                fflush(stdout);
            }
        }
        return ptr;
    } else { // not interested, just let them go via loading the correct
        // Call the original dlopen
        void* ptr = real_dlopen(filename, flags);
        // Print the name of the module being loaded
        if (DL_VERBOSE) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            long long time = ts.tv_nsec + ts.tv_sec * 1e9;
            printf("[info] %lld Loading: %s %p %d\n", time, filename, ptr, flags);
            fflush(stdout);
        }
        return ptr;
    }
}
