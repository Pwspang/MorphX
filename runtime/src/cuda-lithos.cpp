#include "preload.h"
#include "common.h"
#include <cuda.h>   // for cuda related definition

#include <dlfcn.h>
#include <fcntl.h>
#include <execinfo.h>  // for backtrace and backtrace_symbols
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <string>
#include <queue>
#include <deque>
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <algorithm>
#include <unistd.h>
#include <cstdlib>
#include <boost/lockfree/queue.hpp>
#include <boost/lockfree/spsc_queue.hpp>

#define USE_INCREMENTAL

/**
 * Undefine some symbols updated to v2. These are some historical issue with
 * 32bit machine. Now NVIDIA update them to v2 for 64bit.
 */
#undef cuMemAlloc
#undef cuStreamGetCaptureInfo
#undef cuArray3DCreate
#undef cuArray3DGetDescriptor
#undef cuArrayCreate
#undef cuArrayGetDescriptor
#undef cuCtxCreate
#undef cuCtxDestroy
#undef cuCtxPopCurrent
#undef cuCtxPushCurrent
#undef cuDevicePrimaryCtxRelease
#undef cuDevicePrimaryCtxReset
#undef cuDevicePrimaryCtxSetFlags
#undef cuDeviceTotalMem
#undef cuEventDestroy
#undef cuGetProcAddress
#undef cuGraphAddKernelNode
#undef cuGraphExecKernelNodeSetParams
#undef cuGraphExecUpdate
#undef cuGraphicsResourceGetMappedPointer
#undef cuGraphicsResourceSetMapFlags
#undef cuGraphKernelNodeGetParams
#undef cuGraphKernelNodeSetParams
#undef cuIpcOpenMemHandle
#undef cuLinkAddData
#undef cuLinkAddFile
#undef cuLinkCreate
#undef cuMemAllocHost
#undef cuMemAllocPitch
#undef cuMemcpy2DAsync
#undef cuMemcpy2DUnaligned
#undef cuMemcpy2D
#undef cuMemcpy3DAsync
#undef cuMemcpy3D
#undef cuMemcpyAtoA
#undef cuMemcpyAtoD
#undef cuMemcpyAtoHAsync
#undef cuMemcpyAtoH
#undef cuMemcpyDtoA
#undef cuMemcpyDtoDAsync
#undef cuMemcpyDtoD
#undef cuMemcpyDtoHAsync
#undef cuMemcpyDtoH
#undef cuMemcpyHtoAAsync
#undef cuMemcpyHtoA
#undef cuMemcpyHtoDAsync
#undef cuMemcpyHtoD
#undef cuMemFree
#undef cuMemGetAddressRange
#undef cuMemGetInfo
#undef cuMemHostGetDevicePointer
#undef cuMemHostRegister
#undef cuMemsetD16
#undef cuMemsetD2D16
#undef cuMemsetD2D32
#undef cuMemsetD2D8
#undef cuMemsetD32
#undef cuMemsetD8
#undef cuModuleGetGlobal
#undef cuStreamBatchMemOp
#undef cuStreamBeginCapture
#undef cuStreamDestroy
#undef cuStreamWaitValue32
#undef cuStreamWaitValue64
#undef cuStreamWriteValue32
#undef cuStreamWriteValue64
#undef cuTexRefGetAddress
#undef cuTexRefSetAddress2D
#undef cuTexRefSetAddress

#define WARP_SIZE 32 // NVIDIA GPUs use 32 for WARP_SIZE

typedef CUresult (*cuModuleLoadData_func_t)(CUmodule*, const void*);
typedef CUresult (*cuModuleLoadDataEx_func_t)(CUmodule*, const void*, unsigned int, CUjit_option*, void**);
typedef CUresult (*cuModuleGetFunction_func_t)(CUfunction*, CUmodule, const char*);
typedef CUresult (*cuKernelGetFunction_func_t)(CUfunction*, CUkernel);
typedef CUresult (*cuLibraryGetKernel_func_t)(CUkernel*, CUlibrary, const char*);
typedef CUresult (*cuLibraryGetModule_func_t)(CUmodule*, CUlibrary);
typedef CUresult (*cuLibraryLoadData_func_t)(CUlibrary*, const void*, CUjit_option*, void**, unsigned int, CUlibraryOption*, void**, unsigned int);
typedef CUresult (*cuLaunchKernel_func_t)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**);
typedef CUresult (*cuModuleLoad_func_t)(CUmodule*, const char*);
typedef CUresult (*cuModuleLoadFatBinary_func_t)(CUmodule*, const void*);
typedef CUresult (*cuLaunchKernelEx_func_t)(const CUlaunchConfig*, CUfunction, void**, void**);
typedef CUresult (*cuGetProcAddress_func_t)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
typedef CUresult (*cuGetProcAddress_v2_func_t)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);

static cuModuleLoadData_func_t real_cuModuleLoadData = NULL;
static cuModuleLoadDataEx_func_t real_cuModuleLoadDataEx = NULL;
static cuModuleGetFunction_func_t real_cuModuleGetFunction = NULL;
static cuKernelGetFunction_func_t real_cuKernelGetFunction = NULL;
static cuLibraryGetKernel_func_t real_cuLibraryGetKernel = NULL;
static cuLibraryGetModule_func_t real_cuLibraryGetModule = NULL;
static cuLibraryLoadData_func_t real_cuLibraryLoadData = NULL;
static cuLaunchKernel_func_t real_cuLaunchKernel = NULL;
static cuModuleLoad_func_t real_cuModuleLoad = NULL;
static cuModuleLoadFatBinary_func_t real_cuModuleLoadFatBinary = NULL;
static cuLaunchKernelEx_func_t real_cuLaunchKernelEx = NULL;
static cuGetProcAddress_func_t real_cuGetProcAddress = NULL;
static cuGetProcAddress_v2_func_t real_cuGetProcAddress_v2 = NULL;

// helper macro to check cuda error
#define CU_CHECK(call) \
    do { \
        CUresult result = (call); \
        if (result != CUDA_SUCCESS) { \
            const char* errorStr; \
            cuGetErrorString(result, &errorStr); \
            printf("%s failed with error: %d, %s (at %s:%d)\n", #call, result, errorStr, __FILE__, __LINE__); \
            exit(-1); \
        } \
    } while (0)

#define CUDA_CHECK(call) { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}


/////////////////////////////////////////////////////////////////////
// LithOS-style emulation on top of the MorphX pass protocol
/////////////////////////////////////////////////////////////////////
//
// This is NOT LithOS (SOSP'25). It reuses the MorphX compiler pass, which
// turns every kernel into persistent agents that fetch block ids from a global
// counter and only work on SMs in [minSM, maxSM), to approximate two LithOS
// ideas:
//   1. Spatial partitioning: every stream that launches patched kernels gets an
//      SM range (chooseLithosSMRange). With one active stream it gets the whole
//      GPU. With two or more, one stream gets [0, split) and the rest
//      [split, nrSM). Which stream gets [0, split) is set by
//      SMSCHED_LITHOS_RANK: "registration" (first stream to launch a patched
//      kernel; the original behaviour) or "priority" (streams created with a
//      higher CUDA priority, e.g. the LLM under --use-orion).
//   2. Kernel slicing ("atoms"): a kernel runs as a sequence of relaunches of
//      SMSCHED_LITHOS_SLICE_WAVES waves each. A slice is ended by narrowing the
//      kernel's SM range to [0, 0) once enough blocks were fetched; the next
//      slice resumes from the fetched counter and gets a freshly chosen range.
//      So a kernel's partition can change at slice boundaries, and a
//      latency-critical kernel waits at most one slice of the other stream.
//      Ending a slice needs the host: the launching thread polls the progress
//      counter, then waits for the slice to drain before relaunching.
//      SMSCHED_LITHOS_SYNC=0 disables slicing (spatial partitioning only).
// SMSCHED_LITHOS_STEAL=1 (priority mode) lets a stream use the whole GPU for a
// slice when no other stream has a patched kernel in flight, a rough stand-in
// for LithOS's TPC stealing. LithOS's TPC-granular scheduler, hardware
// right-sizing and power management have no equivalent here.
//
// Kernels whose profiled wave time is below 50 us run unmodified, as in
// MorphX's cuda.cpp. Single-slice launches (the common case for small grids)
// are asynchronous; their control words are freed once an event recorded after
// them completes.

using hrc = std::chrono::high_resolution_clock;
using NanoSec = std::chrono::nanoseconds::rep;

struct KernelInfo {
    CUfunction func;
    CUstream stream;
    const char* name;
    int occup;
    int gridSize;

    LaunchSlot slot;  // owns d_agents, fetched|finished, minSM|maxSM
    GdrEntry d_agents;
    GdrEntry fetched;
    GdrEntry finished;
    GdrEntry minSM;
    GdrEntry maxSM;

    int h_minSM;
    int h_maxSM;
};

#define MAX_CONCURRENT_STREAMS 16

template <typename Duration>
inline NanoSec getNano(const Duration &d) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
}

GdrPool* gGdrPool;

struct LithosStream {
    CUstream stream;
    int rank;       // registration order
    int priority;   // CUDA stream priority (lower = more urgent)
    std::atomic<int> inflight{0};  // patched launches not yet known complete
};
std::mutex gStreamMutex;
std::unordered_map<CUstream, LithosStream*> streamInfoMap;
std::atomic<int> nActiveStreams{0};

std::shared_mutex launchMutex;
std::unordered_map<CUfunction, CUfunction> gPkFuncMap;

enum LithosRankMode { RANK_REGISTRATION = 0, RANK_PRIORITY = 1 };

int with_profile;
int profile_log;
int split_hint;
int launch_cache = 1;
int launch_stats = 0;
int lithosSliceWaves;
int lithosSync;
int lithosPollNs;
int lithosVerbose;
int lithosRankMode;
int lithosSteal;
std::vector<std::string> targetKernelPatterns;
int getenv_int(const char* varname, int default_value);

std::atomic<uint64_t> gStatLaunches{0}, gStatSlices{0}, gStatSlicedLaunches{0}, gStatLaunchNs{0};

// Driver queries a patched launch would repeat every time (parameter sizes,
// occupancy, the dynamic shared memory attribute); see cuda.cpp.
struct PkFuncCache {
    std::vector<size_t> argSizes;
    unsigned int maxDynSmem = 0;
    std::unordered_map<uint64_t, int> occupancy;  // blockSize << 32 | smem -> blocks per SM
};
std::mutex gPkCacheMutex;
std::unordered_map<CUfunction, PkFuncCache> gPkCache;

int gNrSM;
inline int getNrSM() {
    if(gNrSM == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        gNrSM = prop.multiProcessorCount;
        if(split_hint == 0) { // SMSCHED_SPLIT_HINT unset
            split_hint = scaleSplitHint(76, gNrSM);
        }
    }
    return gNrSM;
}

inline uint64_t pack4(int x, int y) {
    uint64_t ret = 0;
    ((int*)&ret)[0] = x;
    ((int*)&ret)[1] = y;
    return ret;
}

std::unordered_map<CUstream, std::pair<int, int>> gFixSM;
std::unordered_map<CUstream, std::pair<int, int>> gSuggestSM;
std::unordered_map<CUfunction, std::string> gFuncNameMap;
std::unordered_map<CUkernel, std::string> gKernelNameMap;

std::unordered_map<const char *, float> gCI;
std::unordered_map<const char *, float> gWaveTime;

bool isTargetKernelName(const char* name);

void generateProfileWaves(int nrSM, KernelInfo *info, std::vector<int> &profileWaves, int fixMin = -1, int fixMax = -1) {
    if(fixMin != -1) {
        int gridSize = info->gridSize;
        int wave = (fixMax - fixMin) * info->occup;
        while(gridSize >= wave) {
            profileWaves.push_back(wave);
            gridSize -= wave;
        }
        profileWaves.push_back(gridSize);
        return;
    }
    float waves = 1.0 * info->gridSize / (nrSM * info->occup);
    if(waves <= 1) {
        if(info->gridSize <= 16 * info->occup) {
            profileWaves.push_back(info->gridSize);
        } else {
            profileWaves.push_back(info->gridSize - 8 * info->occup);
            profileWaves.push_back(8 * info->occup);
        }
    } else if(waves < 2) {
        profileWaves.push_back(nrSM * info->occup);
        int left = info->gridSize - nrSM * info->occup;
        if(left <= 16 * info->occup)
            profileWaves.push_back(left);
        else {
            profileWaves.push_back(left - 8 * info->occup);
            profileWaves.push_back(8 * info->occup);
        }
    } else {
        int baseWaves;
        if(waves >= 5) baseWaves = 9;
        else if(waves >= 3.5) baseWaves = 6;
        else if(waves >= 2.5) baseWaves = 4;
        else baseWaves = 3;
        int dec = nrSM / baseWaves;
        // Blocks in one pass over the waves below. Equals the original
        // (dec + nrSM) * occup * baseWaves / 2 when baseWaves divides nrSM;
        // otherwise (e.g. 132 SMs) the loop adds a short last wave.
        int baseBlocks = 0;
        for(int k = nrSM * info->occup; k > 0; k -= dec * info->occup) {
            baseBlocks += k;
        }
        int repeat = info->gridSize / baseBlocks;
        int remain = info->gridSize % baseBlocks;
        for(int k = nrSM * info->occup; k > 0; k -= dec * info->occup) {
            for(int i = 0; i < repeat; ++i) {
                profileWaves.push_back(k);
            }
            if(remain >= k) {
                profileWaves.push_back(k);
                remain -= k;
            }
        }
        if(remain) {
            profileWaves.push_back(remain);
        }
    }
    int sum = 0;
    for(int i = 0; i < profileWaves.size(); ++i) {
        sum += profileWaves[i];
    }
    assert(sum == info->gridSize);
}

inline int getSMFromWave(int wave, int occup) {
    return (wave + occup - 1) / occup;
}

void generateProfileResult(KernelInfo *info, std::vector<std::pair<int, float>> &profileResult) {
    float tm = profileResult[0].second;
    float t0 = profileResult.back().second;
    gWaveTime[info->name] = tm / 1000.0;
    if(profileResult[0].first != getNrSM()) {
        tm = tm / profileResult[0].first * getNrSM();
    }
    float CI = t0 / tm;
    if(CI >= 0.75) CI = 1;
    gCI[info->name] = CI;
    if(profile_log) {
        printf("[MorphX] kernel %.40s: blocks: %d, occup: %d, waves: %d, CI: %lf, WaveTime: %lfus\n", info->name, info->gridSize, info->occup, (int)profileResult.size(), CI, tm / 1000.0);
        for(int i = 0; i < profileResult.size(); ++i) {
            printf("[MorphX] AvgWave %d: %d, %lfus\n", i, profileResult[i].first, profileResult[i].second / 1000.0);
        }
        fflush(stdout);
    }
}

void parseTargetKernelPatterns() {
    targetKernelPatterns.clear();
    const char* raw = std::getenv("SMSCHED_TARGET_PATTERNS");
    if(raw == nullptr || raw[0] == '\0') {
        targetKernelPatterns.emplace_back("blocked_floyd_phase");
        targetKernelPatterns.emplace_back("ggnn5query");
        return;
    }
    const char* start = raw;
    for(const char* p = raw; ; ++p) {
        if(*p == ',' || *p == '\0') {
            if(p > start) {
                targetKernelPatterns.emplace_back(start, p - start);
            }
            if(*p == '\0') break;
            start = p + 1;
        }
    }
}

bool isTargetKernelName(const char* name) {
    if(name == nullptr) return false;
    for(const std::string& pattern : targetKernelPatterns) {
        if(!pattern.empty() && strstr(name, pattern.c_str())) {
            return true;
        }
    }
    return false;
}

inline void writeSM(KernelInfo *info, int minSM, int maxSM) {
    info->h_minSM = minSM;
    info->h_maxSM = maxSM;
    gGdrPool->set(info->minSM, pack4(minSM, maxSM));
}

inline void readProgress(KernelInfo *info, int *fetched, int *finished) {
    uint64_t progress = gGdrPool->get(info->fetched);
    *fetched = ((int*)&progress)[0];
    *finished = ((int*)&progress)[1];
}

void profileThread(KernelInfo *launchInfo, std::vector<int> &profileWaves) {
    if(isTargetKernelName(launchInfo->name)) {
        gCI[launchInfo->name] = 1.0;
        gWaveTime[launchInfo->name] = 100.0;
        return;
    }
    std::vector<std::pair<int, float>> profileTimes;
    int fetched, finished;
    int tot = 0;
    for(int i = 0, j = 0; i < profileWaves.size(); i = j) {
        while(j < profileWaves.size() && profileWaves[j] == profileWaves[i]) {
            tot += profileWaves[j];
            j++;
        }
        auto t0 = hrc::now();
        do {
            readProgress(launchInfo, &fetched, &finished);
            if(fetched >= tot && j < profileWaves.size()) {
                writeSM(launchInfo, 0, getSMFromWave(profileWaves[j], launchInfo->occup));
            }
        } while(finished < tot);
        auto t1 = hrc::now();
        profileTimes.push_back(std::make_pair(getSMFromWave(profileWaves[i], launchInfo->occup), getNano(t1 - t0) / (j - i)));
    }
    generateProfileResult(launchInfo, profileTimes);
}

std::atomic<bool> isInitialized{false};
std::mutex gInitMutex;

extern "C" void fix_SM(CUstream stream, int minSM, int maxSM) {
    gFixSM[stream] = std::make_pair(minSM, maxSM);
}

extern "C" void suggest_SM(CUstream stream, int minSM, int maxSM) {
    gSuggestSM[stream] = std::make_pair(minSM, maxSM);
}

int getenv_int(const char* varname, int default_value) {
    const char* val_str = std::getenv(varname);
    if (val_str != nullptr) {
        return std::atoi(val_str);
    }
    return default_value;
}

// Does any stream other than `self` have a patched launch in flight?
bool othersInflight(const LithosStream *self) {
    std::lock_guard<std::mutex> lock(gStreamMutex);
    for(auto &kv : streamInfoMap) {
        if(kv.second != self && kv.second->inflight.load(std::memory_order_acquire) > 0) {
            return true;
        }
    }
    return false;
}

// Is `self` the stream that gets [0, split)?
bool isFirstPartition(const LithosStream *self) {
    if(lithosRankMode == RANK_PRIORITY) {
        std::lock_guard<std::mutex> lock(gStreamMutex);
        int best = self->priority;
        bool tie = false;
        for(auto &kv : streamInfoMap) {
            if(kv.second == self) continue;
            if(kv.second->priority < best) return false;
            if(kv.second->priority == best) tie = true;
        }
        // All streams equally urgent: fall back to registration order.
        if(!tie) return true;
    }
    return self->rank == 0;
}

std::pair<int, int> chooseLithosSMRange(const LithosStream *self, int nrSM) {
    if(nrSM <= 1) {
        return {0, nrSM};
    }
    if(gFixSM.count(self->stream)) {
        auto p = gFixSM[self->stream];
        int minSM = std::clamp(p.first, 0, nrSM);
        int maxSM = std::clamp(p.second, minSM, nrSM);
        if(maxSM > minSM) {
            return {minSM, maxSM};
        }
    }
    if(nActiveStreams.load() <= 1) {
        return {0, nrSM};
    }
    if(lithosSteal && !othersInflight(self)) {
        return {0, nrSM};
    }
    int split = std::clamp(split_hint, 1, nrSM - 1);
    if(isFirstPartition(self)) {
        return {0, split};
    }
    return {split, nrSM};
}

/////////////////////////////////////////////////////////////////////////
// Below is the implementation of dynamic loader and function hooking
/////////////////////////////////////////////////////////////////////////

#if defined(__cplusplus)
extern "C" {
#endif

// include auto-generated signatures for unmodified functions
// @note signature.c will be auto generated by parse.py
#include "signature.c"

/**
 * Function to initialize the environment, including
 * * init the cuda driver module via dlopen
 * * init the file system as specified above
 * * init the hashmap for binaries and CUfunction
 * * init commonly used functions like real_cuModuleLoad...
 *
 * @note this will be called only once when any hooked driver function is called
 */
static void ld_init(void) {
    pthread_once(&mutex_is_initialized, mutex_init);
    // init() is critical section to be protected
    pthread_mutex_lock(&mutex);
    if (shared_lib != NULL) { // then it has been initialized by another
        pthread_mutex_unlock(&mutex);
        return;
    }
    with_profile = getenv_int("SMSCHED_PROFILE", 1);
    profile_log = getenv_int("SMSCHED_PROFILE_LOG", 0);
    launch_cache = getenv_int("MORPHX_LAUNCH_CACHE", 1);
    launch_stats = getenv_int("MORPHX_LAUNCH_STATS", 0);
    // 0 = default: 76 on a 108-SM A100, scaled to the SM count in getNrSM()
    split_hint = getenv_int("SMSCHED_SPLIT_HINT", 0);
    lithosSliceWaves = std::max(1, getenv_int("SMSCHED_LITHOS_SLICE_WAVES", 1));
    lithosSync = getenv_int("SMSCHED_LITHOS_SYNC", 1);
    lithosPollNs = std::max(0, getenv_int("SMSCHED_LITHOS_POLL_NS", 0));
    lithosVerbose = getenv_int("SMSCHED_LITHOS_VERBOSE", 0);
    const char *rank = getenv("SMSCHED_LITHOS_RANK");
    lithosRankMode = (rank && strcmp(rank, "priority") == 0) ? RANK_PRIORITY : RANK_REGISTRATION;
    lithosSteal = getenv_int("SMSCHED_LITHOS_STEAL", 0);
    parseTargetKernelPatterns();
    fprintf(stderr, "[smsched][lithos] profile=%d split_hint=%d (0 = 76 scaled to the SM count) slice_waves=%d sync=%d rank=%s steal=%d poll_ns=%d verbose=%d\n",
        with_profile, split_hint, lithosSliceWaves, lithosSync,
        lithosRankMode == RANK_PRIORITY ? "priority" : "registration", lithosSteal, lithosPollNs, lithosVerbose);
    common_init(); // init common modules
    real_cuModuleLoadData      = (cuModuleLoadData_func_t) dlsym(shared_lib, "cuModuleLoadData");
    real_cuModuleLoadDataEx    = (cuModuleLoadDataEx_func_t) dlsym(shared_lib, "cuModuleLoadDataEx");
    real_cuModuleGetFunction   = (cuModuleGetFunction_func_t) dlsym(shared_lib, "cuModuleGetFunction");
    real_cuKernelGetFunction   = (cuKernelGetFunction_func_t) dlsym(shared_lib, "cuKernelGetFunction");
    real_cuLibraryGetKernel    = (cuLibraryGetKernel_func_t) dlsym(shared_lib, "cuLibraryGetKernel");
    real_cuLibraryGetModule    = (cuLibraryGetModule_func_t) dlsym(shared_lib, "cuLibraryGetModule");
    real_cuLibraryLoadData     = (cuLibraryLoadData_func_t) dlsym(shared_lib, "cuLibraryLoadData");
    real_cuLaunchKernel        = (cuLaunchKernel_func_t) dlsym(shared_lib, "cuLaunchKernel");
    real_cuModuleLoad          = (cuModuleLoad_func_t) dlsym(shared_lib, "cuModuleLoad");
    real_cuModuleLoadFatBinary = (cuModuleLoadFatBinary_func_t) dlsym(shared_lib, "cuModuleLoadFatBinary");
    real_cuLaunchKernelEx      = (cuLaunchKernelEx_func_t) dlsym(shared_lib, "cuLaunchKernelEx");
    real_cuGetProcAddress      = (cuGetProcAddress_func_t) dlsym(shared_lib, "cuGetProcAddress");
    real_cuGetProcAddress_v2    = (cuGetProcAddress_v2_func_t) dlsym(shared_lib, "cuGetProcAddress_v2");
    init_unmodified(); // init unmodified functions, defined in signature.c
    CHECK_DL(); // checking if any dl error presented
    // leaving critical section, unlock
    pthread_mutex_unlock(&mutex);
    return;
}

static void runtimeShutdown();

static void runtimeInit() {
    std::lock_guard<std::mutex> lock(gInitMutex);
    if(isInitialized.load()) return;
    gGdrPool = new GdrPool();
    std::atexit(runtimeShutdown);
    isInitialized.store(true);
}

static LithosStream* checkStream(CUstream stream) {
    if(stream == 0) {
        std::cerr << "Default stream is not supported" << std::endl;
        exit(EXIT_FAILURE);
    }
    if(!isInitialized.load()) {
        runtimeInit();
    }
    std::lock_guard<std::mutex> lock(gStreamMutex);
    auto it = streamInfoMap.find(stream);
    if(it != streamInfoMap.end()) {
        return it->second;
    }
    if(nActiveStreams.load() >= MAX_CONCURRENT_STREAMS) {
        std::cerr << "Too many streams" << std::endl;
        exit(EXIT_FAILURE);
    }
    LithosStream *info = new LithosStream();
    info->stream = stream;
    info->rank = nActiveStreams.load();
    info->priority = 0;
    real_cuStreamGetPriority(stream, &info->priority);
    streamInfoMap[stream] = info;
    nActiveStreams.fetch_add(1);
    // One line per stream: shows which stream got [0, split).
    fprintf(stderr, "[smsched][lithos] register stream=%p rank=%d priority=%d\n", stream, info->rank, info->priority);
    return info;
}

// Launches whose control words must outlive the call: freed once the event
// recorded after their last slice has completed.
struct PendingFree {
    CUevent done;
    LaunchSlot slot;
    LithosStream *stream;
};
static std::mutex gPendingMutex;
static std::deque<PendingFree> gPending;
static std::vector<CUevent> gEventPool;

static CUevent takeEvent() {
    // caller holds gPendingMutex
    if(!gEventPool.empty()) {
        CUevent ev = gEventPool.back();
        gEventPool.pop_back();
        return ev;
    }
    CUevent ev;
    CU_CHECK(real_cuEventCreate(&ev, CU_EVENT_DISABLE_TIMING));
    return ev;
}

static void reapPending(bool wait) {
    std::lock_guard<std::mutex> lock(gPendingMutex);
    while(!gPending.empty()) {
        PendingFree &p = gPending.front();
        CUresult q = wait ? real_cuEventSynchronize(p.done) : real_cuEventQuery(p.done);
        if(q == CUDA_ERROR_NOT_READY) {
            break;
        }
        gGdrPool->launch_free(p.slot);
        p.stream->inflight.fetch_sub(1, std::memory_order_acq_rel);
        gEventPool.push_back(p.done);
        gPending.pop_front();
    }
}

static void deferFree(KernelInfo *info, LithosStream *stream) {
    std::lock_guard<std::mutex> lock(gPendingMutex);
    CUevent ev = takeEvent();
    CU_CHECK(real_cuEventRecord(ev, info->stream));
    gPending.push_back({ev, info->slot, stream});
}

static void runtimeShutdown() {
    if(!isInitialized.load()) return;
    reapPending(true);
    if(launch_stats) {
        uint64_t n = gStatLaunches.load();
        // Wall time of a patched launch call, including host waits between slices.
        fprintf(stderr, "[smsched][lithos-stats] patched_launches=%lu sliced_launches=%lu slices=%lu us_per_launch_call=%.2f\n",
            (unsigned long)n, (unsigned long)gStatSlicedLaunches.load(), (unsigned long)gStatSlices.load(),
            n ? gStatLaunchNs.load() / 1000.0 / n : 0.0);
    }
}

CUresult cuModuleLoadData(CUmodule* module, const void* image) {
    if (shared_lib == NULL) {
        ld_init();
    }
    CUresult result = real_cuModuleLoadData(module, image);
    return result;
}

CUresult cuLibraryLoadData(CUlibrary* library, const void* code, CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, CUlibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) {
    if (shared_lib == NULL) {
        ld_init();
    }
    CUresult result = real_cuLibraryLoadData(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);
    return result;
}

CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues) {
    if (shared_lib == NULL) {
        ld_init();
    }
    CUresult ret = real_cuModuleLoadDataEx(module, image, numOptions, options, optionValues);
    return ret;
}

// JAX use this API, but they don't pass in fatbin but cubin, so a wrong API to use...
CUresult cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin) {
    if (shared_lib == NULL) {
        ld_init();
    }
    CUresult result = real_cuModuleLoadFatBinary(module, fatCubin); // call the symbol
    return result;
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    if (shared_lib == NULL) {
        ld_init();
    }
    CUresult result = real_cuModuleGetFunction(hfunc, hmod, name);
    std::string pk_name = std::string(name) + "_pk";
    CUfunction pk_func;
    CUresult pk_result = real_cuModuleGetFunction(&pk_func, hmod, pk_name.c_str());
    gFuncNameMap[*hfunc] = name;
    if(pk_result == CUDA_SUCCESS) {
        gPkFuncMap[*hfunc] = pk_func;
        gFuncNameMap[pk_func] = pk_name;
    } else {
        gPkFuncMap[*hfunc] = nullptr;
    }
    return result;
}

CUresult cuKernelGetFunction(CUfunction* pFunc, CUkernel kernel) {
    if (shared_lib == NULL) {
        ld_init();
    }
    CUresult result = real_cuKernelGetFunction(pFunc, kernel);
    auto nameIter = gKernelNameMap.find(kernel);
    if(result == CUDA_SUCCESS && nameIter != gKernelNameMap.end()) {
        gFuncNameMap[*pFunc] = nameIter->second;
        // lazy loading (cuLibrary*) never calls cuModuleGetFunction, so look up
        // the patched "_pk" kernel in the function's module here
        CUmodule mod;
        if(!gPkFuncMap.count(*pFunc) && real_cuFuncGetModule(&mod, *pFunc) == CUDA_SUCCESS) {
            std::string pk_name = nameIter->second + "_pk";
            CUfunction pk_func;
            if(real_cuModuleGetFunction(&pk_func, mod, pk_name.c_str()) == CUDA_SUCCESS) {
                gPkFuncMap[*pFunc] = pk_func;
                gFuncNameMap[pk_func] = pk_name;
            } else {
                gPkFuncMap[*pFunc] = nullptr;
            }
        }
    }
    return result;
}

CUresult cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, const char* name) {
    if (shared_lib == NULL) {
        ld_init();
    }
    CUresult result = real_cuLibraryGetKernel(pKernel, library, name);
    if(result == CUDA_SUCCESS && name != nullptr) {
        gKernelNameMap[*pKernel] = name;
    }
    return result;
}

CUresult cuLibraryGetModule(CUmodule* pMod, CUlibrary library) {
    if (shared_lib == NULL) {
        ld_init();
    }
    CUresult result = real_cuLibraryGetModule(pMod, library);
    return result;
}

/**
 * cuLaunchKernelEx attributes that cuLaunchKernel can drop without changing
 * what the kernel computes. Anything else (thread block clusters, cooperative
 * launch, launch events, ...) needs the original launch, e.g. Hopper CUTLASS 3.
 */
static bool hasRequiredLaunchAttrs(const CUlaunchConfig* config) {
    for(unsigned int i = 0; i < config->numAttrs; ++i) {
        switch(config->attrs[i].id) {
        case CU_LAUNCH_ATTRIBUTE_IGNORE:
        case CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW:
        case CU_LAUNCH_ATTRIBUTE_PRIORITY:
        case CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION:
        case CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP:
        case CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN:
            break;
        default:
            return true;
        }
    }
    return false;
}

// Parameter sizes and occupancy of a patched function; see PkFuncCache.
static const std::vector<size_t>& queryPkLaunch(CUfunction f, int blockSize, unsigned int sharedMemBytes, int *occup) {
    static thread_local std::vector<size_t> uncached;
    if(!launch_cache) {
        CU_CHECK(real_cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, sharedMemBytes));
        CU_CHECK(real_cuOccupancyMaxActiveBlocksPerMultiprocessor(occup, f, blockSize, sharedMemBytes));
        uncached.clear();
        size_t offset, size;
        for(int i = 0; real_cuFuncGetParamInfo(f, i, &offset, &size) == CUDA_SUCCESS; i++) {
            uncached.push_back(size);
        }
        return uncached;
    }
    std::lock_guard<std::mutex> lock(gPkCacheMutex);
    PkFuncCache &cache = gPkCache[f];
    if(cache.argSizes.empty()) {
        size_t offset, size;
        for(int i = 0; real_cuFuncGetParamInfo(f, i, &offset, &size) == CUDA_SUCCESS; i++) {
            cache.argSizes.push_back(size);
        }
    }
    if(sharedMemBytes > cache.maxDynSmem) {
        CU_CHECK(real_cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, sharedMemBytes));
        cache.maxDynSmem = sharedMemBytes;
    }
    uint64_t key = ((uint64_t)blockSize << 32) | sharedMemBytes;
    auto it = cache.occupancy.find(key);
    if(it == cache.occupancy.end()) {
        int blocks = 0;
        CU_CHECK(real_cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, f, blockSize, sharedMemBytes));
        it = cache.occupancy.emplace(key, blocks).first;
    }
    *occup = it->second;
    return cache.argSizes;
}

/**
 * Execution Control, cuLaunchXXX and cuFuncXXX
 * @see https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html
 */
CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
    CUstream hStream, void** kernelParams, void** extra)
{
    if (shared_lib == NULL) {
        ld_init();
    }

    // cudart may pass a CUkernel as the CUfunction; use its context function
    if(gKernelNameMap.count((CUkernel)f)) {
        CUfunction kernelFunc;
        if(cuKernelGetFunction(&kernelFunc, (CUkernel)f) == CUDA_SUCCESS) {
            f = kernelFunc;
        }
    }

    int nrSM = getNrSM();
    int gridSize = gridDimX * gridDimY * gridDimZ;

    const char* funcName = nullptr;
    auto funcNameIter = gFuncNameMap.find(f);
    if(funcNameIter != gFuncNameMap.end()) {
        funcName = funcNameIter->second.c_str();
    } else {
        CU_CHECK(real_cuFuncGetName(&funcName, f));
        if(funcName != nullptr) {
            gFuncNameMap[f] = funcName;
            funcName = gFuncNameMap[f].c_str();
        }
    }

    int inWhiteList = isTargetKernelName(funcName) ? 1 : 0;
    if (!inWhiteList && gWaveTime.count(funcName) && gWaveTime[funcName] < 50.0) {
        return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    }
    if(inWhiteList && !gCI.count(funcName)) {
        gCI[funcName] = 1.0;
        gWaveTime[funcName] = 100.0;
    }

    CUfunction newF = gPkFuncMap[f];
    if(newF == nullptr) {
        return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    }
    f = newF;
    auto tLaunch = hrc::now();
    LithosStream *streamInfo = checkStream(hStream);
    reapPending(false);
    sharedMemBytes = std::max(sharedMemBytes, 4u);
    int blockSize = blockDimX * blockDimY * blockDimZ;
    int occup = 0;
    const std::vector<size_t> &argSizes = queryPkLaunch(f, blockSize, sharedMemBytes, &occup);
    assert(occup > 0);
    int numArgs = argSizes.size();
    assert(numArgs >= 6);

    KernelInfo *launchInfo = new KernelInfo;
    launchInfo->func = f;
    launchInfo->stream = hStream;
    launchInfo->name = funcName;
    launchInfo->occup = occup;
    launchInfo->gridSize = gridSize;
    launchInfo->slot = gGdrPool->launch_alloc();
    launchInfo->d_agents = launchInfo->slot.agents;
    launchInfo->fetched = launchInfo->slot.fetched;
    launchInfo->finished = launchInfo->fetched.half();
    launchInfo->minSM = launchInfo->slot.minSM;
    launchInfo->maxSM = launchInfo->minSM.half();

    // The driver copies parameter values at launch, so the original pointers
    // can be passed for every slice; only the six protocol arguments are ours.
    void **nargs = new void*[numArgs];
    for(int i = 0; i < numArgs - 6; ++i) {
        nargs[i] = kernelParams[i];
    }
    dim3 gridDim(gridDimX, gridDimY, gridDimZ);
    nargs[numArgs - 6] = &gridDim;
    nargs[numArgs - 5] = &launchInfo->d_agents.d;
    nargs[numArgs - 4] = &launchInfo->fetched.d;
    nargs[numArgs - 3] = &launchInfo->finished.d;
    nargs[numArgs - 2] = &launchInfo->minSM.d;
    nargs[numArgs - 1] = &launchInfo->maxSM.d;

    // One agent per block slot on every SM; agents on SMs outside the slice's
    // range exit at once, so every SM in the range runs at full occupancy.
    int agents = occup * nrSM;
    CUresult res = CUDA_SUCCESS;
    bool deferred = false;
    if(with_profile && !gCI.count(launchInfo->name)) {
        std::unique_lock<std::shared_mutex> writeLock(launchMutex);
        int fixMin = -1, fixMax = -1;
        if(gFixSM.count(hStream)) {
            auto p = gFixSM[hStream];
            fixMin = p.first; fixMax = p.second;
        }
        std::vector<int> profileWaves;
        generateProfileWaves(nrSM, launchInfo, profileWaves, fixMin, fixMax);
        launchInfo->h_minSM = 0;
        launchInfo->h_maxSM = getSMFromWave(profileWaves[0], occup);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        gGdrPool->launch_init(launchInfo->slot, pack4(agents, 0), pack4(0, launchInfo->h_maxSM), true, hStream);
        std::thread profile(profileThread, launchInfo, std::ref(profileWaves));
        res = real_cuLaunchKernel(f, agents, 1, 1, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, nargs, extra);
        profile.join();
        if(res == CUDA_SUCCESS) {
            CU_CHECK(real_cuStreamSynchronize(hStream));
        }
    } else {
        std::shared_lock<std::shared_mutex> readLock(launchMutex);
        streamInfo->inflight.fetch_add(1, std::memory_order_acq_rel);
        int fetched = 0, finished = 0;
        int sliceId = 0;
        while(finished < gridSize) {
            auto smRange = chooseLithosSMRange(streamInfo, nrSM);
            int minSM = smRange.first;
            int maxSM = smRange.second;
            int sliceBlocks = std::max(1, (maxSM - minSM) * occup) * lithosSliceWaves;
            int sliceEnd = lithosSync ? std::min(gridSize, fetched + sliceBlocks) : gridSize;
            bool lastSlice = sliceEnd >= gridSize;
            launchInfo->h_minSM = minSM;
            launchInfo->h_maxSM = maxSM;
            // Stream-ordered (memcpy: copy + event wait on hStream); a slice after
            // a cut keeps fetched|finished and overrides the [0, 0) range.
            gGdrPool->launch_init(launchInfo->slot, pack4(agents, 0), pack4(minSM, maxSM), sliceId == 0, hStream);
            if(lithosVerbose) {
                fprintf(stderr, "[smsched][lithos] launch %.40s slice=%d start=%d end=%d grid=%d sm=[%d,%d) agents=%d stream=%p\n",
                    launchInfo->name, sliceId, fetched, sliceEnd, gridSize, minSM, maxSM, agents, hStream);
            }
            res = real_cuLaunchKernel(f, agents, 1, 1, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, nargs, extra);
            gStatSlices.fetch_add(1, std::memory_order_relaxed);
            if(res != CUDA_SUCCESS) {
                break;
            }
            if(lastSlice) {
                // Nothing to cut: let it run and free the control words later.
                deferFree(launchInfo, streamInfo);
                deferred = true;
                break;
            }
            // End the slice once its blocks have been handed out; agents finish
            // the block they hold, see [0, 0) and exit.
            while(true) {
                readProgress(launchInfo, &fetched, &finished);
                if(fetched >= sliceEnd || finished >= gridSize) {
                    break;
                }
                if(lithosPollNs > 0) {
                    std::this_thread::sleep_for(std::chrono::nanoseconds(lithosPollNs));
                }
            }
            writeSM(launchInfo, 0, 0);
            CU_CHECK(real_cuStreamSynchronize(hStream));
            // Also drains the control stream, so the cut has landed before the
            // next slice's range is written.
            readProgress(launchInfo, &fetched, &finished);
            if(lithosVerbose) {
                fprintf(stderr, "[smsched][lithos] done %.40s slice=%d target=%d fetched=%d finished=%d overshoot=%d\n",
                    launchInfo->name, sliceId, sliceEnd, std::min(fetched, gridSize), std::min(finished, gridSize),
                    std::max(0, std::min(fetched, gridSize) - sliceEnd));
            }
            sliceId++;
        }
        if(sliceId > 0) {
            gStatSlicedLaunches.fetch_add(1, std::memory_order_relaxed);
        }
        if(!deferred) {
            streamInfo->inflight.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
    if(launch_stats) {
        gStatLaunches.fetch_add(1, std::memory_order_relaxed);
        gStatLaunchNs.fetch_add(getNano(hrc::now() - tLaunch), std::memory_order_relaxed);
    }
    if(!deferred) {
        gGdrPool->launch_free(launchInfo->slot);
    }
    delete launchInfo;
    delete[] nargs;
    CU_CHECK(res);
    return res;
}

CUresult cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {
    if (shared_lib == NULL) {
        ld_init();
    }
    if(config == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if(hasRequiredLaunchAttrs(config)) {
        return real_cuLaunchKernelEx(config, f, kernelParams, extra);
    }
    return cuLaunchKernel(
        f,
        config->gridDimX,
        config->gridDimY,
        config->gridDimZ,
        config->blockDimX,
        config->blockDimY,
        config->blockDimZ,
        config->sharedMemBytes,
        config->hStream,
        kernelParams,
        extra);
}

CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
    CUstream hStream, void** kernelParams, void** extra) {
    return cuLaunchKernel(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}

CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {
    return cuLaunchKernelEx(config, f, kernelParams, extra);
}

/**
 * Following functions shall also be hooked but we don't observe any workload
 * calling them, thus having a [info] section for tracing, add if needed
 */
CUresult cuModuleLoad(CUmodule* module, const char* fname) {
    if (shared_lib == NULL) {
        ld_init();
    }
    CUresult result = real_cuModuleLoad(module, fname); // call the symbol
    return result;
}

#define cuLaunchKernel_ptsz cuLaunchKernel_ptsz_unmodified
#define cuLaunchKernelEx_ptsz cuLaunchKernelEx_ptsz_unmodified
#include "unmodified.c" // include the auto-generated code
#undef cuLaunchKernel_ptsz
#undef cuLaunchKernelEx_ptsz

CUresult cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus) {
    if (shared_lib == NULL) {
        ld_init();
    }
    std::string sym_str(symbol);
    void *hook = nullptr;
    if (sym_str == "cuGetProcAddress") hook = (void*)cuGetProcAddress_v2;
    else if (sym_str == "cuLaunchKernel") hook = (void*)cuLaunchKernel;
    else if (sym_str == "cuLaunchKernel_ptsz") hook = (void*)cuLaunchKernel_ptsz;
    else if (sym_str == "cuModuleLoadData") hook = (void*)cuModuleLoadData;
    else if (sym_str == "cuLibraryLoadData") hook = (void*)cuLibraryLoadData;
    else if (sym_str == "cuModuleLoadDataEx") hook = (void*)cuModuleLoadDataEx;
    else if (sym_str == "cuModuleGetFunction") hook = (void*)cuModuleGetFunction;
    else if (sym_str == "cuKernelGetFunction") hook = (void*)cuKernelGetFunction;
    else if (sym_str == "cuLibraryGetKernel") hook = (void*)cuLibraryGetKernel;
    else if (sym_str == "cuLibraryGetModule") hook = (void*)cuLibraryGetModule;
    else if (sym_str == "cuModuleLoadFatBinary") hook = (void*)cuModuleLoadFatBinary;
    else if (sym_str == "cuLaunchKernelEx") hook = (void*)cuLaunchKernelEx;
    else if (sym_str == "cuLaunchKernelEx_ptsz") hook = (void*)cuLaunchKernelEx_ptsz;
    if (hook != nullptr) {
        *pfn = hook;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    }
    CUresult result = real_cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus);
    return result;
}

CUresult cuGetProcAddress(const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus) {
    if (shared_lib == NULL) {
        ld_init();
    }
    std::string sym_str(symbol);
    void *hook = nullptr;
    if (sym_str == "cuGetProcAddress") hook = (void*)cuGetProcAddress;
    else if (sym_str == "cuLaunchKernel") hook = (void*)cuLaunchKernel;
    else if (sym_str == "cuLaunchKernel_ptsz") hook = (void*)cuLaunchKernel_ptsz;
    else if (sym_str == "cuModuleLoadData") hook = (void*)cuModuleLoadData;
    else if (sym_str == "cuLibraryLoadData") hook = (void*)cuLibraryLoadData;
    else if (sym_str == "cuModuleLoadDataEx") hook = (void*)cuModuleLoadDataEx;
    else if (sym_str == "cuModuleGetFunction") hook = (void*)cuModuleGetFunction;
    else if (sym_str == "cuKernelGetFunction") hook = (void*)cuKernelGetFunction;
    else if (sym_str == "cuLibraryGetKernel") hook = (void*)cuLibraryGetKernel;
    else if (sym_str == "cuLibraryGetModule") hook = (void*)cuLibraryGetModule;
    else if (sym_str == "cuModuleLoadFatBinary") hook = (void*)cuModuleLoadFatBinary;
    else if (sym_str == "cuLaunchKernelEx") hook = (void*)cuLaunchKernelEx;
    else if (sym_str == "cuLaunchKernelEx_ptsz") hook = (void*)cuLaunchKernelEx_ptsz;
    if (hook != nullptr) {
        *pfn = hook;
        return CUDA_SUCCESS;
    }
    CUresult result = real_cuGetProcAddress(symbol, pfn, cudaVersion, flags, symbolStatus);
    return result;
}

#ifdef __cplusplus
}
#endif
