/*
 * cuGetProcAddress interposition, shared by the high- and low-priority
 * libraries.
 *
 * CUDA >= 11.3 runtimes (including the cudart linked into cuBLAS) resolve
 * every driver entry point through cuGetProcAddress, so an exported wrapper is
 * only reached when this hands it out. The real driver is asked first, so the
 * requested CUDA version and per-thread-stream flags select the right variant
 * and symbolStatus is the driver's own; a wrapper replaces the result only when
 * the driver returned exactly the function that wrapper wraps.
 *
 * TGS_TRACE_GPA=1 logs every lookup. TGS_GPA_LEGACY=1 restores the original
 * behaviour (answer hooked names directly, whatever version or flags).
 * TGS_GPA_HOOKS=name,name,... limits which names are hooked ("none" = pure
 * pass-through); meant for bisecting.
 */
#include <string.h>

#include "../include/cuda-helper.h"
#include "../include/hijack.h"

extern entry_t cuda_library_entry[];

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
                          cuuint64_t flags);
CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                             cuuint64_t flags,
                             CUdriverProcAddressQueryResult *symbolStatus);
CUresult cuLaunchKernel();
CUresult cuLaunchKernel_ptsz();
CUresult cuLaunchKernelEx();
CUresult cuLaunchKernelEx_ptsz();
CUresult cuLaunchCooperativeKernel();
CUresult cuLaunchCooperativeKernel_ptsz();
CUresult cuModuleLoadData();
CUresult cuLibraryLoadData();
CUresult cuModuleLoadDataEx();
CUresult cuModuleLoadFatBinary();
CUresult cuModuleGetFunction();
CUresult cuKernelGetFunction();
CUresult cuLibraryGetKernel();
CUresult cuLibraryGetModule();

typedef struct {
  const char *name;
  cuda_entry_enum_t real;
  void *hook;
} gpa_hook_t;

static const gpa_hook_t g_hooks[] = {
    {"cuGetProcAddress", CUDA_ENTRY_ENUM(cuGetProcAddress), (void *)cuGetProcAddress},
    {"cuGetProcAddress", CUDA_ENTRY_ENUM(cuGetProcAddress_v2), (void *)cuGetProcAddress_v2},
    {"cuLaunchKernel", CUDA_ENTRY_ENUM(cuLaunchKernel), (void *)cuLaunchKernel},
    {"cuLaunchKernel", CUDA_ENTRY_ENUM(cuLaunchKernel_ptsz), (void *)cuLaunchKernel_ptsz},
    {"cuLaunchKernelEx", CUDA_ENTRY_ENUM(cuLaunchKernelEx), (void *)cuLaunchKernelEx},
    {"cuLaunchKernelEx", CUDA_ENTRY_ENUM(cuLaunchKernelEx_ptsz), (void *)cuLaunchKernelEx_ptsz},
    {"cuLaunchCooperativeKernel", CUDA_ENTRY_ENUM(cuLaunchCooperativeKernel), (void *)cuLaunchCooperativeKernel},
    {"cuLaunchCooperativeKernel", CUDA_ENTRY_ENUM(cuLaunchCooperativeKernel_ptsz), (void *)cuLaunchCooperativeKernel_ptsz},
    {"cuModuleLoadData", CUDA_ENTRY_ENUM(cuModuleLoadData), (void *)cuModuleLoadData},
    {"cuLibraryLoadData", CUDA_ENTRY_ENUM(cuLibraryLoadData), (void *)cuLibraryLoadData},
    {"cuModuleLoadDataEx", CUDA_ENTRY_ENUM(cuModuleLoadDataEx), (void *)cuModuleLoadDataEx},
    {"cuModuleLoadFatBinary", CUDA_ENTRY_ENUM(cuModuleLoadFatBinary), (void *)cuModuleLoadFatBinary},
    {"cuModuleGetFunction", CUDA_ENTRY_ENUM(cuModuleGetFunction), (void *)cuModuleGetFunction},
    {"cuKernelGetFunction", CUDA_ENTRY_ENUM(cuKernelGetFunction), (void *)cuKernelGetFunction},
    {"cuLibraryGetKernel", CUDA_ENTRY_ENUM(cuLibraryGetKernel), (void *)cuLibraryGetKernel},
    {"cuLibraryGetModule", CUDA_ENTRY_ENUM(cuLibraryGetModule), (void *)cuLibraryGetModule},
};
#define GPA_HOOKS (sizeof(g_hooks) / sizeof(g_hooks[0]))

static int env_flag(const char *name) {
  const char *v = getenv(name);
  return v && v[0] && strcmp(v, "0") != 0;
}

static int hook_enabled(const char *name) {
  static const char *list = (const char *)-1;
  size_t n = strlen(name);
  const char *p;

  if (list == (const char *)-1) list = getenv("TGS_GPA_HOOKS");
  if (!list) return 1;
  for (p = list; (p = strstr(p, name)) != NULL; p += n) {
    if ((p == list || p[-1] == ',') && (p[n] == ',' || p[n] == '\0')) return 1;
  }
  return 0;
}

static CUresult resolve(const char *symbol, void **pfn, int cudaVersion,
                        cuuint64_t flags,
                        CUdriverProcAddressQueryResult *symbolStatus, int v2) {
  static int trace = -1, legacy = -1;
  CUresult ret;
  void *real = NULL;
  const char *swapped = "";
  size_t i;

  load_necessary_data();
  if (trace < 0) trace = env_flag("TGS_TRACE_GPA");
  if (legacy < 0) legacy = env_flag("TGS_GPA_LEGACY");

  if (legacy) {
    for (i = 0; i < GPA_HOOKS; i++) {
      if (strcmp(symbol, g_hooks[i].name) == 0) {
        /* The original code always answered the first hook of a name. */
        *pfn = g_hooks[i].hook;
        if (v2 && symbolStatus) *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        if (trace)
          fprintf(stderr, "[TGS-GPA] %s ver=%d flags=%llu -> legacy hook\n",
                  symbol, cudaVersion, (unsigned long long)flags);
        return CUDA_SUCCESS;
      }
    }
  }

  if (v2) {
    ret = CUDA_ENTRY_CALL(cuda_library_entry, cuGetProcAddress_v2, symbol, pfn,
                          cudaVersion, flags, symbolStatus);
  } else {
    ret = CUDA_ENTRY_CALL(cuda_library_entry, cuGetProcAddress, symbol, pfn,
                          cudaVersion, flags);
  }
  if (ret == CUDA_SUCCESS && pfn && *pfn) {
    real = *pfn;
    for (i = 0; i < GPA_HOOKS; i++) {
      if (real == cuda_library_entry[g_hooks[i].real].fn_ptr &&
          hook_enabled(g_hooks[i].name)) {
        *pfn = g_hooks[i].hook;
        swapped = cuda_library_entry[g_hooks[i].real].name;
        break;
      }
    }
  }
  if (trace)
    fprintf(stderr, "[TGS-GPA] %s%s ver=%d flags=%llu -> ret=%d status=%d%s%s\n",
            symbol, v2 ? "" : " (v1)", cudaVersion, (unsigned long long)flags,
            (int)ret, (v2 && symbolStatus) ? (int)*symbolStatus : -1,
            swapped[0] ? " hooked=" : "", swapped);
  return ret;
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                             cuuint64_t flags,
                             CUdriverProcAddressQueryResult *symbolStatus) {
  return resolve(symbol, pfn, cudaVersion, flags, symbolStatus, 1);
}

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
                          cuuint64_t flags) {
  return resolve(symbol, pfn, cudaVersion, flags, NULL, 0);
}
