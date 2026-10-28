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
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <array>
#include <vector>
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
            morphx_fatal_exit(-1); \
        } \
    } while (0)

#define CUDA_CHECK(call) { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        morphx_fatal_exit(); \
    } \
}

/////////////////////////////////////////////////////////////////////
// Below is the implementation of SM scheduler
/////////////////////////////////////////////////////////////////////

using hrc = std::chrono::high_resolution_clock;
using NanoSec = std::chrono::nanoseconds::rep;

struct KernelInfo {
    CUfunction func;
    CUstream stream;
    const char* name;
    int occup;
    int gridSize;
    dim3 logicalGrid;

    LaunchSlot slot;  // owns d_agents, fetched|finished, minSM|maxSM
    GdrEntry d_agents;
    GdrEntry fetched;
    GdrEntry finished;
    GdrEntry minSM;
    GdrEntry maxSM;

    int h_minSM;
    int h_maxSM;

    int isScheduled;

    std::atomic<float>* ci;   // &gCI[name] once profiled: the agent thread reads this, never the map
    struct KernelStats* stats; // per-kernel counters (SMSCHED_PROFILE_LOG=1), else nullptr
};

// SMSCHED_PROFILE_LOG=1: per kernel name (not per profile key), what the runtime did with its launches.
struct KernelStats {
    std::atomic<long> launches{0}, shortBypass{0}, profiled{0}, managed{0}, ciLimited{0}, moved{0};
};

struct StreamInfo {
    CUstream stream;
    boost::lockfree::spsc_queue<KernelInfo*> *pending;
    hrc::time_point tStart;
};

enum KernelType {
    MEMORY_INTENSIVE = 0,
    COMPUTE_INTENSIVE = 1
};

// Kernel classification from the profile's CI (t0 / tm, see generateProfileResult).
// Defaults are the A100 values; MORPHX_CI_THRESHOLD / MORPHX_CI_SATURATE retune them.
float ci_threshold = 0.9f;
float ci_saturate = 0.75f;
int ci_mode_occ = 0; // MORPHX_CI_MODE=occ, see generateProfileResult

KernelType getKernelType(float CI) {
    if(CI >= ci_threshold) return COMPUTE_INTENSIVE;
    return MEMORY_INTENSIVE;
}

// #define PROFILE_ITERATION
// #define NO_QUEUEING
#define MAX_CONCURRENT_STREAMS 16

template <typename Duration>
inline NanoSec getNano(const Duration &d) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
}

GdrPool* gGdrPool;

std::unordered_map<CUstream, StreamInfo*> streamInfoMap;
StreamInfo *streamInfos[MAX_CONCURRENT_STREAMS];
std::atomic<int> nActiveStreams{0};
std::mutex launchMutex;
int getenv_int(const char* varname, int default_value);

std::thread agent;
std::atomic<bool> gAgentRunning{true};

std::unordered_map<CUfunction, CUmodule> gFuncModuleMap;
std::unordered_map<CUkernel, bool> gKernelSet;

int with_profile;
int profile_log;
std::atomic<long> gCILimitedLaunches{0};  // CI kernels limited to [0, split) at launch
std::atomic<long> gSplitMoves{0};         // co-runners moved to [split, nrSM) by schedule_2
std::atomic<long> gManagedLaunches{0};    // transformed (_pk) launches outside profiling
int split_hint;
int llm_trick_mode;
int llm_trace_limit;
std::atomic<int> llm_trace_prints{0};
std::unordered_map<std::string, int> llm_trace_seen;

int gNrSM;
inline int getNrSM() {
    if(gNrSM == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        gNrSM = prop.multiProcessorCount;
        if(split_hint == 0) { // SMSCHED_SPLIT_HINT unset
            split_hint = scaleSplitHint(60, gNrSM);
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

// std::unordered_map<const char *, std::vector<std::pair<int, float>>> gProfileData; // TODO: thread unsafe
std::unordered_map<const char *, std::atomic<float>> gCI;
std::unordered_map<const char *, std::atomic<float>> gWaveTime;

// MORPHX_PROFILE_PER_GRID=1 (default off): profile, classify (CI) and short-bypass per
// (kernel name, grid-size bucket) instead of per name. Per name, the first launch decides
// for the whole run; LLM kernels whose first launch is a small warm-up batch (e.g. the
// decode attention kernel) are then short-bypassed forever. Buckets are powers of two of
// the logical grid size; at most MORPHX_PROFILE_GRID_MAX (default 6) buckets get their own
// profile per name, later ones reuse the nearest profiled bucket, so the profiling cost
// stays bounded. The key is an interned "<name>#g<bucket>" string, so gCI, gWaveTime and
// the short-kernel check share it, and hasPattern() on it still matches the name.
int profile_per_grid = 0;
int profile_grid_max = 6;
int short_bypass_total = 0;  // MORPHX_SHORT_BYPASS_TOTAL, see generateProfileResult
std::unordered_map<const char *, std::array<const char *, 33>> gGridKeys; // launch thread only
std::unordered_map<const char *, KernelStats*> gKernelStats;              // launch thread only

const char* profileKey(const char* name, int gridSize) {
    if(!profile_per_grid) return name;
    int b = gridSize <= 1 ? 0 : 32 - __builtin_clz((unsigned)(gridSize - 1)); // ceil(log2)
    auto it = gGridKeys.find(name);
    if(it == gGridKeys.end()) {
        std::array<const char *, 33> none;
        none.fill(nullptr);
        it = gGridKeys.emplace(name, none).first;
    }
    auto &keys = it->second;
    if(keys[b]) return keys[b];
    int used = 0, nearest = -1;
    for(int i = 0; i <= 32; ++i) {
        if(!keys[i]) continue;
        ++used;
        if(nearest < 0 || std::abs(i - b) < std::abs(nearest - b)) nearest = i;
    }
    if(used >= profile_grid_max && nearest >= 0) return keys[nearest];
    char suffix[24];
    snprintf(suffix, sizeof(suffix), "#g%u", 1u << std::min(b, 31));
    std::string key = std::string(name) + suffix;
    keys[b] = strdup(key.c_str());
    return keys[b];
}

KernelStats* kernelStats(const char* name) {
    if(!profile_log) return nullptr;
    auto &st = gKernelStats[name];
    if(!st) st = new KernelStats();
    return st;
}

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
    // printf("[smsched] kernel %s: gridSize: %d, occup: %d, nrSM: %d, waves: %lf\n", info->name, info->gridSize, info->occup, nrSM, waves);
    // fflush(stdout);
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
        // printf("[smsched] baseWaves: %d, dec: %d, baseBlocks: %d, repeat: %d, remain: %d\n", baseWaves, dec, baseBlocks, repeat, remain);
        // printf("[smsched] nrSM: %d, occup: %d\n", nrSM, info->occup);
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

void generateProfileResult(KernelInfo *info, std::vector<int> &profileWaves, std::vector<NanoSec> &profileTimes) {
    if(profileWaves.size() != profileTimes.size()) {
        fprintf(stderr, "Profile size mismatch: %lu vs %lu\n", profileWaves.size(), profileTimes.size());
        exit(EXIT_FAILURE);
    }
    std::vector<std::pair<int, float>> profileResult;
    std::vector<int> profileBlocks; // blocks per wave of each profileResult entry
    for(int l = 0, r; l < profileWaves.size(); l = r) {
        float avg = 0;
        for(r = l; r < profileWaves.size() && profileWaves[r] == profileWaves[l]; ++r) {
            avg += profileTimes[r];
        }
        avg /= (r - l);
        profileResult.push_back({getSMFromWave(profileWaves[l], info->occup), avg});
        profileBlocks.push_back(profileWaves[l]);
    }
    float tm = profileResult[0].second;
    float t0 = profileResult.back().second;
    // The 50 us short-kernel bypass compares this value. Default: the first profile
    // wave's time. MORPHX_SHORT_BYPASS_TOTAL=1: that wave scaled to the whole grid,
    // i.e. the kernel's estimated run time; kernels made of many short blocks (the
    // non-persistent FA3 prefill: ~16 us per 132-block wave, ~300 us per launch) are
    // then managed instead of bypassed.
    float waveUs = tm / 1000.0;
    if(short_bypass_total && profileWaves[0] > 0)
        waveUs *= (float)info->gridSize / (float)profileWaves[0];
    gWaveTime[info->name].store(waveUs);
    // legacy (A100 default): last wave vs the first wave scaled up to all SMs.
    float legacyTm = tm;
    if(profileResult[0].first != getNrSM()) {
        legacyTm = tm / profileResult[0].first * getNrSM();
    }
    float legacyCI = t0 / legacyTm;
    // occ: per-wave time with fewer SMs vs with more SMs, both waves at full
    // occupancy (blocks == SMs * occup) and excluding the kernel's last wave (its
    // tail: the GPU drains, so it runs faster than a steady-state wave).
    // Compute-bound -> ~1, bandwidth-bound -> < 1. Only kernels with at least two
    // such waves keep the whole GPU busy long enough to be worth confining to
    // [0, split); smaller grids (e.g. decode GEMMs, one partial wave) get CI 0.
    // On H200 the legacy formula fails both ways: its SM-count scaling marks any
    // grid under one wave as memory-bound regardless of how it scales, and the
    // fast tail wave pulls multi-wave prefill GEMMs down to CI ~0.4.
    int fullFirst = -1, fullLast = -1;
    size_t nSteady = profileResult.size() > 1 ? profileResult.size() - 1 : profileResult.size();
    for(size_t i = 0; i < nSteady; ++i) {
        if(profileBlocks[i] == profileResult[i].first * info->occup) {
            if(fullFirst < 0) fullFirst = i;
            fullLast = i;
        }
    }
    float occCI = 0;
    if(fullFirst >= 0 && fullLast > fullFirst) {
        occCI = profileResult[fullLast].second / profileResult[fullFirst].second;
    }
    float CI = ci_mode_occ ? occCI : legacyCI;
    float rawCI = CI;
    if(CI >= ci_saturate) CI = 1;
    gCI[info->name].store(CI);
    if(profile_log) {
        std::string waves;
        for(size_t i = 0; i < profileResult.size(); ++i) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%s%d/%d:%.1f", waves.empty() ? "" : ",", profileResult[i].first,
                profileBlocks[i], profileResult[i].second / 1000.0);
            waves += buf;
        }
        fprintf(stderr, "[MorphX-profile] name=%.120s stream=%p grid=%dx%dx%d occup=%d waves=%s ci_legacy=%.3f ci_occ=%.3f ci_raw=%.3f ci=%.3f ci_kernel=%d\n",
            info->name, (void*)info->stream, info->logicalGrid.x, info->logicalGrid.y, info->logicalGrid.z, info->occup,
            waves.c_str(), legacyCI, occCI, rawCI, CI, CI >= ci_threshold);
    }
    // printf("[smsched] kernel %s: CI: %lf, WaveTime: %lfus\n", info->name, CI, tm / 1000.0);
    // for(int i = 0; i < profileResult.size(); ++i) {
    //     printf("[smsched] AvgWave %d: %d, %lfus\n", i, profileResult[i].first, profileResult[i].second / 1000.0);
    // }

}

void bind_thread_to_cpu(int k) {
    pinThreadToAllowedCpu(k);
}
std::atomic<int> gCpu;

inline void writeSM(KernelInfo *info, int minSM, int maxSM) {
    // if(maxSM >= info->h_maxSM && minSM <= info->h_minSM)
    //     return;
    info->h_minSM = minSM;
    info->h_maxSM = maxSM;
    gGdrPool->set(info->minSM, pack4(minSM, maxSM));
}

void profileThread(KernelInfo *launchInfo, std::vector<int> &profileWaves) {
    const char* name = launchInfo->name;
    // if(strstr(name, "flash_fwd_kernel")) {
    //     gCI[name].store(1.0);
    //     return;
    // }
    if(strstr(name, "at_cuda_detail")) {
        gCI[name].store(0.5);
        gWaveTime[name].store(1.0);
        return;
    }
    std::vector<NanoSec> profileTimes;
    int fetched, finished;
    int tot = 0;
    for(int i = 0; i < profileWaves.size(); ++i) {
        tot += profileWaves[i];
        auto t0 = hrc::now();
        do {
            uint64_t tmp = gGdrPool->get(launchInfo->fetched);
            fetched = ((int*)&tmp)[0];
            finished = ((int*)&tmp)[1];
            // printf("[smsched][profile]: wave %d/%lu, fetched: %d, finished: %d, tot: %d\n",
            //     i + 1, profileWaves.size(), fetched, finished, tot);
            if(fetched >= tot && i + 1 < profileWaves.size() && profileWaves[i+1] != profileWaves[i]) {
                writeSM(launchInfo, 0, getSMFromWave(profileWaves[i+1], launchInfo->occup));
            }
        } while(finished < tot);
        auto t1 = hrc::now();
        profileTimes.push_back(getNano(t1 - t0));
    }
    generateProfileResult(launchInfo, profileWaves, profileTimes);
}

bool isCI(KernelInfo *info) {
    float CI = info->ci ? info->ci->load() : gCI[info->name].load();
    if(CI >= ci_threshold) return true;
    return false;
}

int getStreamRank(CUstream stream) {
    for(int i = 0; i < nActiveStreams.load(); ++i) {
        if(streamInfos[i]->stream == stream)
            return i;
    }
    return -1;
}

void schedule_1(std::vector<KernelInfo*> &actives, std::vector<KernelInfo*> &pendings, int nrSM) {
    // for (auto info : actives) {
    //     if(strstr(info->name, "redLoadIteratorMixedINS1S_1")) {
    //         writeSM(info, 0, 80);
    //     }
    // }
    for (auto info : pendings) {
        if(strstr(info->name, "redLoadIteratorMixedINS1S_1")) {
            for (auto ainfo : actives) {
                writeSM(ainfo, 80, nrSM);
                // printf("[smsched] Adjusting in stream %p\n", ainfo->stream);
            }
            for (auto pinfo : pendings) {
                if(pinfo != info) {
                    writeSM(pinfo, 0, 80);
                    // printf("[smsched] Adjusting in stream %p\n", pinfo->stream);
                }
            }
        }
    }
}

void schedule(std::vector<KernelInfo*> &actives, std::vector<KernelInfo*> &pendings, int nrSM) {
    schedule_1(actives, pendings, nrSM);
    return;

    if(actives.empty())
        return; // Wait for the hardware to choose the next kernel to execute.
    if(actives.size() + pendings.size() <= 1)
        return; // Only one stream, no need to schedule.
    if(actives.size() >= 2)
        return; // Either we have achieved optimal configuration by previous scheduling, or it's kernel boundaries.
    KernelInfo *info0 = actives[0], *info1 = nullptr;
    if(info0->h_minSM != 0 || info0->h_maxSM != nrSM)
        return; // The active kernel has already been limited by SM, wait for it to finish.
    // printf("[smsched] scheduling... gCI = %f, actives[0] = %s\n", gCI[info0->name].load(), info0->name);
    fflush(stdout);
    float CI0 = gCI[info0->name].load();
    //// special strategy
    if (CI0 > 0.9) {
        // printf("[smsched] Adjusting in stream %p\n", info0->stream);
        writeSM(info0, 0, 75);
    }
    return;
    //// end
    for (auto p : pendings) {
        float CI1 = gCI[p->name].load();
        if(getKernelType(CI0) != getKernelType(CI1)) {
            info1 = p;
        }
    }
    if(info1 == nullptr)
        return; // No suitable kernel to co-locate.
    if(info1->h_minSM != 0) {
        writeSM(info0, 0, nrSM - info1->h_minSM);
        return;
    }
    if(info1->h_maxSM != nrSM) {
        writeSM(info0, info1->h_maxSM, nrSM);
        return;
    }
    float CI1 = gCI[info1->name].load();
    if(CI0 > 0.9) CI0 = 3.0;
    if(CI1 > 0.9) CI1 = 3.0;
    int div = (1 - CI0) / (CI1 - CI0) * nrSM;
    // printf("[smsched] scheduling kernels: (CI: %lf) and (CI: %lf), div: %d\n", CI0, CI1, div);
    assert(div > 0 && div < nrSM);
    writeSM(info0, 0, div);
    writeSM(info1, div, nrSM);
}

void schedule_2(KernelInfo *kernels[], KernelInfo *nexts[], int nStreams, int nrSM) {
    if(split_hint == 0)
        split_hint = 60;
    for(int i = 0; i < nStreams; ++i) {
        if(kernels[i] && isCI(kernels[i])) {
            // if(!nexts[i]->isScheduled) {
            //     writeSM(nexts[i], 0, split_hint);
            //     nexts[i]->isScheduled = 1;
            // }
            for(int j = 0; j < nStreams; ++j) {
                if(j != i && kernels[j] && !kernels[j]->isScheduled) {
                    writeSM(kernels[j], split_hint, nrSM);
                    kernels[j]->isScheduled = 1;
                    gSplitMoves.fetch_add(1, std::memory_order_relaxed);
                    if(kernels[j]->stats) kernels[j]->stats->moved.fetch_add(1, std::memory_order_relaxed);
                }
                if(j != i && nexts[j] && !nexts[j]->isScheduled) {
                    writeSM(nexts[j], split_hint, nrSM);
                    nexts[j]->isScheduled = 1;
                    gSplitMoves.fetch_add(1, std::memory_order_relaxed);
                    if(nexts[j]->stats) nexts[j]->stats->moved.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }

    }
}

void agentThread() {
    morphx_in_agent_thread = true;
    bind_thread_to_cpu(gCpu.fetch_add(1));
    int nrSM = getNrSM();
    KernelInfo *kernels[MAX_CONCURRENT_STREAMS], *nexts[MAX_CONCURRENT_STREAMS];
    memset(kernels, 0, sizeof(kernels));
    memset(nexts, 0, sizeof(nexts));
    // std::vector<KernelInfo*> actives;
    // std::vector<KernelInfo*> pendings;
    int cnt = 0;
    cudaStream_t tStream;
    CUDA_CHECK(cudaStreamCreate(&tStream));
    // Read every stream's fetched|finished with one get_many (one control-stream
    // sync with the memcpy backend) instead of one get() per stream.
    bool batchReads = getenv_int("MORPHX_AGENT_BATCH", 1) != 0;
    GdrEntry reads[MAX_CONCURRENT_STREAMS];
    uint64_t values[MAX_CONCURRENT_STREAMS];
    int readStream[MAX_CONCURRENT_STREAMS];
    while(gAgentRunning.load(std::memory_order_relaxed)) {
        int nStreams = nActiveStreams.load();
        int nReads = 0;
        // actives.clear();
        // pendings.clear();
        for(int i = 0; i < nStreams; i++) {
            if(i == 0)
                cnt = (cnt + 1) % 500;
            if(nexts[i] == nullptr) {
                streamInfos[i]->pending->pop(nexts[i]);
            }
            if(kernels[i] == nullptr) {
                kernels[i] = nexts[i];
                nexts[i] = nullptr;
                streamInfos[i]->pending->pop(nexts[i]);
            }
            if(kernels[i] == nullptr) {
                continue;
            }
            readStream[nReads] = i;
            reads[nReads++] = kernels[i]->fetched;
        }
        if(batchReads && nReads > 0) {
            gGdrPool->get_many(reads, values, nReads);
        }
        for(int r = 0; r < nReads; r++) {
            int i = readStream[r];
            uint64_t tmp = batchReads ? values[r] : gGdrPool->get(kernels[i]->fetched);
            int fetched = ((int*)&tmp)[0];
            int finished = ((int*)&tmp)[1];
            if(cnt == 0) {
                // printf("[smsched] stream %p, fetched: %d, finished: %d, gridSize: %d, kernels[i]: %p, nexts[i]: %p, func: %p\n",
                //     kernels[i]->stream, fetched, finished, kernels[i]->gridSize, kernels[i], nexts[i], kernels[i]->func);
                // int agents;
                // CUDA_CHECK(cudaMemcpyAsync(&agents, kernels[i]->d_agents, sizeof(int), cudaMemcpyDeviceToHost, tStream));
                // printf("[smsched] stream %p, agents: %d\n", kernels[i]->stream, agents);
            }
            assert(fetched >= finished);
            if(finished >= kernels[i]->gridSize) {
                gGdrPool->launch_free(kernels[i]->slot);
                delete kernels[i];
                kernels[i] = nullptr;
                continue;
            }
            // if(fetched == 0) {
            //     pendings.push_back(kernels[i]);
            // } else {
            //     actives.push_back(kernels[i]);
            // }

        }
        schedule_2(kernels, nexts, nStreams, nrSM);
        // schedule(actives, pendings, nrSM);
    }
}

StreamInfo* registerStream(CUstream stream) {
    // printf("Register stream %p\n", stream);
    StreamInfo *streamInfo = new StreamInfo();
    streamInfo->stream = stream;
    streamInfo->pending = new boost::lockfree::spsc_queue<KernelInfo*>(65536);
    streamInfoMap[stream] = streamInfo;
    if(nActiveStreams.load() >= 16) {
        std::cerr << "Too many streams" << std::endl;
        exit(EXIT_FAILURE);
    }
    streamInfos[nActiveStreams.load()] = streamInfo;
    nActiveStreams.fetch_add(1);
    return streamInfo;
}

bool isInitialized = false;

// SMSCHED_PROFILE_LOG=1: how many managed launches ran on [0, split) (CI),
// were moved to [split, nrSM) next to a CI kernel, or kept the whole GPU.
// Exported so a host that leaves with os._exit (test-pd-online.py) can call it.
extern "C" void morphx_report() {
    static std::atomic<bool> reported{false};
    if(!profile_log || reported.exchange(true)) return;
    long managed = gManagedLaunches.load(), ci = gCILimitedLaunches.load(), moved = gSplitMoves.load();
    fprintf(stderr, "[MorphX-profile] summary ci_threshold=%.3f ci_saturate=%.3f split=%d nrSM=%d"
        " managed_launches=%ld ci_limited_launches=%ld split_moves=%ld full_gpu_launches=%ld\n",
        ci_threshold, ci_saturate, split_hint, gNrSM, managed, ci, moved, std::max(0L, managed - ci - moved));
    if(profile_per_grid) {
        size_t keys = 0;
        for(auto &kv : gGridKeys) for(auto k : kv.second) keys += k != nullptr;
        fprintf(stderr, "[MorphX-profile] per-grid profiling: %zu names, %zu (name, grid bucket) keys, max %d per name\n",
            gGridKeys.size(), keys, profile_grid_max);
    }
    // Per kernel: patched (_pk) kernels and anything with managed launches, by managed count.
    std::vector<std::pair<const char*, KernelStats*>> rows(gKernelStats.begin(), gKernelStats.end());
    std::sort(rows.begin(), rows.end(), [](auto &a, auto &b) { return a.second->managed.load() > b.second->managed.load(); });
    for(auto &r : rows) {
        KernelStats *s = r.second;
        if(s->managed.load() == 0 && s->profiled.load() == 0) continue;
        fprintf(stderr, "[MorphX-profile] kernel launches=%ld short_bypass=%ld profiled=%ld managed=%ld ci_limited=%ld moved=%ld name=%.140s\n",
            s->launches.load(), s->shortBypass.load(), s->profiled.load(), s->managed.load(), s->ciLimited.load(),
            s->moved.load(), r.first);
    }
    fflush(stderr);
}

// Ensure the background agent thread is properly joined before process exit
static void runtimeShutdown() {
    if(!isInitialized) return;
    morphx_report();
    gAgentRunning.store(false, std::memory_order_relaxed);
    if(agent.joinable()) {
        agent.join();
    }
}

void runtimeInit() {
    agent = std::thread(agentThread);
    gGdrPool = new GdrPool();
    // Register graceful shutdown to avoid std::terminate at process exit
    std::atexit(runtimeShutdown);
    isInitialized = true;
}

extern "C" void fix_SM(CUstream stream, int minSM, int maxSM) {
    gFixSM[stream] = std::make_pair(minSM, maxSM);
    // printf("fix SM for stream %p: %d %d\n", stream, minSM, maxSM);
}

extern "C" void suggest_SM(CUstream stream, int minSM, int maxSM) {
    gSuggestSM[stream] = std::make_pair(minSM, maxSM);
    // printf("suggest SM for stream %p: %d %d\n", stream, minSM, maxSM);
}

inline StreamInfo* checkStream(CUstream stream) {
    if(stream == 0) {
        std::cerr << "Default stream is not supported" << std::endl;
        exit(EXIT_FAILURE);
    }
    if(!isInitialized) {
        runtimeInit();
    }
    auto streamInfoIter = streamInfoMap.find(stream);
    if(streamInfoIter == streamInfoMap.end()) {
        return registerStream(stream);
    } else {
        return streamInfoIter->second;
    }
}

int getenv_int(const char* varname, int default_value) {
    const char* val_str = std::getenv(varname);
    if (val_str != nullptr) {
        return std::atoi(val_str);
    }
    return default_value;
}

bool hasPattern(const char* name, const char* pattern) {
    return strstr(name, pattern) != nullptr;
}

bool isTargetLLMKernel(const char* name) {
    bool legacyTarget = hasPattern(name, "NSL_IfLi4ELb1EEESF_EENS1R_25Til")
        || hasPattern(name, "redLoadIteratorMixedINS1S_1")
        || hasPattern(name, "BatchDecode");
    if(legacyTarget) {
        return true;
    }
    if(llm_trick_mode < 10 || llm_trick_mode > 13) {
        return false;
    }
    return hasPattern(name, "cutlassL6Kernel")
        || hasPattern(name, "BatchPrefillWithPagedKVCacheKernel")
        || hasPattern(name, "BatchDecodeWithPagedKVCacheKernel")
        || hasPattern(name, "PersistentVariableLengthMergeStatesKernel");
}

bool useTargetOnlyMode() {
    return llm_trick_mode == 1 || llm_trick_mode == 3 || llm_trick_mode == 6 || llm_trick_mode == 7
        || llm_trick_mode == 10 || llm_trick_mode == 11 || llm_trick_mode == 14 || llm_trick_mode == 15;
}

bool usePostLaunchSpecialMode() {
    return llm_trick_mode == 2 || llm_trick_mode == 3 || llm_trick_mode == 5 || llm_trick_mode == 7
        || llm_trick_mode == 13;
}

bool keepShortTargetKernelsMode() {
    return llm_trick_mode == 4 || llm_trick_mode == 5 || llm_trick_mode == 6 || llm_trick_mode == 7
        || llm_trick_mode == 11 || llm_trick_mode == 12 || llm_trick_mode == 13 || llm_trick_mode == 15;
}

bool disableLaunchTimeCILimitMode() {
    return llm_trick_mode == 14 || llm_trick_mode == 15 || llm_trick_mode == 16;
}

void traceLLMKernel(
    const char* phase,
    const char* name,
    int gridSize,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream stream) {
    if(llm_trace_limit <= 0) {
        return;
    }
    std::string key = std::string(phase) + ":" + name;
    int &seen = llm_trace_seen[key];
    seen++;
    if(seen > 1) {
        return;
    }
    int index = llm_trace_prints.fetch_add(1);
    if(index >= llm_trace_limit) {
        return;
    }
    fprintf(
        stderr,
        "[MorphX-llm] phase=%s grid=%d block=(%u,%u,%u) shared=%u stream=%p name=%s\n",
        phase,
        gridSize,
        blockDimX,
        blockDimY,
        blockDimZ,
        sharedMemBytes,
        stream,
        name);
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
    if(const char *v = std::getenv("MORPHX_CI_THRESHOLD")) ci_threshold = std::strtof(v, nullptr);
    if(const char *v = std::getenv("MORPHX_CI_SATURATE")) ci_saturate = std::strtof(v, nullptr);
    if(const char *v = std::getenv("MORPHX_CI_MODE")) ci_mode_occ = strcmp(v, "occ") == 0;
    // 0 = default: 60 on a 108-SM A100, scaled to the SM count in getNrSM()
    split_hint = getenv_int("SMSCHED_SPLIT_HINT", 0);
    llm_trick_mode = getenv_int("SMSCHED_LLM_TRICK_MODE", 0);
    llm_trace_limit = getenv_int("SMSCHED_LLM_TRACE_LIMIT", 0);
    llm_trace_prints.store(0);
    profile_per_grid = getenv_int("MORPHX_PROFILE_PER_GRID", 0);
    profile_grid_max = std::max(1, std::min(33, getenv_int("MORPHX_PROFILE_GRID_MAX", 6)));
    short_bypass_total = getenv_int("MORPHX_SHORT_BYPASS_TOTAL", 0);
    // Profiles are inserted while the agent thread runs; keep the maps from rehashing.
    gCI.reserve(4096);
    gWaveTime.reserve(4096);
    // fprintf(stderr, "[smsched] with_profile=%d, split_hint=%d\n", with_profile, split_hint);
    common_init(); // init common modules
    // load hooked function of Neutrino
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
    // initialzie the L2 Flush Memory if benchmark is enabled
    // fprintf(event_log, "[info] init success\n");
    // leaving critical section, unlock
    pthread_mutex_unlock(&mutex);
    return;
}


CUresult cuModuleLoadData(CUmodule* module, const void* image) {
    if (shared_lib == NULL) {
        // fprintf(event_log, "[mod] cuModuleLoadData called before ld_init, calling ld_init now...\n");
        ld_init();
    }
    // fprintf(stderr, "[mod] cuModuleLoadData called for image %p\n", image);
    CUresult result = real_cuModuleLoadData(module, image);
    return result;
}

CUresult cuLibraryLoadData(CUlibrary* library, const void* code, CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, CUlibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) {
    if (shared_lib == NULL) {
        // fprintf(event_log, "[mod] cuLibraryLoadData called before ld_init, calling ld_init now...\n");
        ld_init();
    }
    // fprintf(stderr, "[mod] cuLibraryLoadData called for code %p\n", code);
    CUresult result = real_cuLibraryLoadData(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);
    return result;
}

CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues) {
    if (shared_lib == NULL) {
        // fprintf(event_log, "[mod] cuModuleLoadDataEx called before ld_init, calling ld_init now...\n");
        ld_init();
    }
    // fprintf(stderr, "[mod] cuModuleLoadDataEx called for image %p\n", image);
    CUresult ret = real_cuModuleLoadDataEx(module, image, numOptions, options, optionValues);
    return ret;
}

// JAX use this API, but they don't pass in fatbin but cubin, so a wrong API to use...
CUresult cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin) {
    if (shared_lib == NULL) {
        // fprintf(event_log, "[mod] cuModuleLoadFatBinary called before ld_init, calling ld_init now...\n");
        ld_init();
    }
    // fprintf(stderr, "[mod] cuModuleLoadFatBinary called for fatCubin %p\n", fatCubin);
    CUresult result = real_cuModuleLoadFatBinary(module, fatCubin); // call the symbol
    return result;
}

// @todo handle the multiple function with different name problem
CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    if (shared_lib == NULL) {
        // fprintf(event_log, "[mod] cuModuleGetFunction called before ld_init, calling ld_init now...\n");
        ld_init();
    }
    // fprintf(stderr, "[mod] cuModuleGetFunction called for %s\n", name);
    CUresult result = real_cuModuleGetFunction(hfunc, hmod, name);
    gFuncModuleMap[*hfunc] = hmod;
    return result;
}

CUresult cuKernelGetFunction(CUfunction* pFunc, CUkernel kernel) {
    if (shared_lib == NULL) {
        // fprintf(event_log, "[mod] cuKernelGetFunction called before ld_init, calling ld_init now...\n");
        ld_init();
    }
    // fprintf(stderr, "[mod] cuKernelGetFunction called for %p\n", kernel);
    CUresult result = real_cuKernelGetFunction(pFunc, kernel);
    // lazy loading (cuLibrary*) never calls cuModuleGetFunction; record the
    // module so cuLaunchKernel can find the patched "_pk" kernel
    CUmodule mod;
    if(result == CUDA_SUCCESS && real_cuFuncGetModule(&mod, *pFunc) == CUDA_SUCCESS) {
        gFuncModuleMap[*pFunc] = mod;
    }
    return result;
}

CUresult cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, const char* name) {
    // fprintf(stderr, "[mod] cuLibraryGetKernel called for %s\n", name);
    if (shared_lib == NULL) {
        // fprintf(event_log, "[mod] cuLibraryGetKernel called before ld_init, calling ld_init now...\n");
        ld_init();
    }
    CUresult result = real_cuLibraryGetKernel(pKernel, library, name);
    if(result == CUDA_SUCCESS) {
        gKernelSet[*pKernel] = true;
    }
    return result;
}

CUresult cuLibraryGetModule(CUmodule* pMod, CUlibrary library) {
    if (shared_lib == NULL) {
        // fprintf(event_log, "[mod] cuLibraryGetModule called before ld_init, calling ld_init now...\n");
        ld_init();
    }
    // fprintf(stderr, "[mod] cuLibraryGetModule called for library %p\n", library);
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

/**
 * Execution Control, cuLaunchXXX and cuFuncXXX
 * @see https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html
 *
 * aims at providing runtime probing support
 */

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
    CUstream hStream, void** kernelParams, void** extra)
{
    launchMutex.lock();

    if (shared_lib == NULL) {
        // fprintf(event_log, "[mod] cuLaunchKernel called before ld_init, calling ld_init now...\n");
        ld_init();
    }

    // cudart may pass a CUkernel as the CUfunction; use its context function
    if(gKernelSet.count((CUkernel)f)) {
        CUfunction kernelFunc;
        if(cuKernelGetFunction(&kernelFunc, (CUkernel)f) == CUDA_SUCCESS) {
            f = kernelFunc;
        }
    }

    int nrSM = getNrSM();
    int gridSize = gridDimX * gridDimY * gridDimZ;

    const char* funcName;
    CU_CHECK(real_cuFuncGetName(&funcName, f));

    if(useTargetOnlyMode() && !isTargetLLMKernel(funcName)) {
        traceLLMKernel("target-bypass", funcName, gridSize, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream);
        launchMutex.unlock();
        return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    }

    // Profile key: the name, or (name, grid bucket) with MORPHX_PROFILE_PER_GRID=1.
    const char* profKey = profileKey(funcName, gridSize);
    KernelStats* kstats = kernelStats(funcName);
    if(kstats) kstats->launches.fetch_add(1, std::memory_order_relaxed);
    if ((gWaveTime.count(profKey) && gWaveTime[profKey].load() < 50.0)
        && !(keepShortTargetKernelsMode() && isTargetLLMKernel(funcName))) {
        if(kstats) kstats->shortBypass.fetch_add(1, std::memory_order_relaxed);
        traceLLMKernel("short-bypass", funcName, gridSize, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream);
        launchMutex.unlock();
        return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    }

    if(strstr(funcName, "at6native")) {
        traceLLMKernel("native-bypass", funcName, gridSize, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream);
        launchMutex.unlock();
        return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    }

    if(strstr(funcName, "float32")) {
        launchMutex.unlock();
        return CUDA_SUCCESS;
        return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    }

    if(0
        // || strstr(funcName, "NSL_IfLi4ELb1EEESF_EENS1R_25Til")
        // || strstr(funcName, "redLoadIteratorMixedINS1S_1")
        // || strstr(funcName, "BatchDecode")
        // || strstr(funcName, "_SF_SD_SX_fSF_NST_13OpMul")
        // || strstr(funcName, "14_S17_fSF_EENS1I_18Shar")
    ) {
        launchMutex.unlock();
        return CUDA_SUCCESS;
    }

    // if(gCI.count(funcName) && gCI[funcName].load() >= 0.99) {
    //     launchMutex.unlock();
    //     return CUDA_SUCCESS;
    // }

    std::string newFuncName(funcName);
    newFuncName += "_pk";
    CUfunction newF;
    CUmodule mod = gFuncModuleMap[f];
    CUresult tres = real_cuModuleGetFunction(&newF, mod, newFuncName.c_str());
    if(tres != CUDA_SUCCESS) {
        // fprintf(stderr, "[smsched] cannot find kernel %s, use original kernel\n", newFuncName.c_str());
        traceLLMKernel("unpatched", funcName, gridSize, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream);
        launchMutex.unlock();
        return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    } else {
        // printf("[smsched] using patched kernel %s\n", newFuncName.c_str());
        traceLLMKernel("patched", funcName, gridSize, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream);
        f = newF;
    }
    StreamInfo *streamInfo = checkStream(hStream);
    sharedMemBytes = std::max(sharedMemBytes, 4u);
    CU_CHECK(real_cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, sharedMemBytes));
    int blockSize = blockDimX * blockDimY * blockDimZ;
    KernelInfo *launchInfo = new KernelInfo;
    launchInfo->func = f;
    launchInfo->stream = hStream;
    launchInfo->name = profKey;
    launchInfo->isScheduled = 0;
    launchInfo->ci = nullptr;
    launchInfo->stats = kstats;
    CU_CHECK(real_cuOccupancyMaxActiveBlocksPerMultiprocessor(&launchInfo->occup, f, blockSize, sharedMemBytes));
    assert(launchInfo->occup > 0);
    std::vector<size_t> argSizes;
    int i = 0;
    size_t offset, size;
    while(real_cuFuncGetParamInfo(f, i, &offset, &size) == CUDA_SUCCESS) {
        argSizes.push_back(size);
        i++;
    }
    int numArgs = argSizes.size();
    void **nargs = new void*[numArgs];
    for(int i = 0; i < numArgs - 6; ++i) {
        nargs[i] = malloc(argSizes[i]);
        memcpy(nargs[i], kernelParams[i], argSizes[i]);
    }

    launchInfo->gridSize = gridSize;
    launchInfo->logicalGrid = dim3(gridDimX, gridDimY, gridDimZ);
    // Written once, right before the launch, by launch_init().
    launchInfo->slot = gGdrPool->launch_alloc();
    launchInfo->d_agents = launchInfo->slot.agents;
    launchInfo->fetched = launchInfo->slot.fetched;
    launchInfo->finished = launchInfo->fetched.half();
    launchInfo->minSM = launchInfo->slot.minSM;
    launchInfo->maxSM = launchInfo->minSM.half();
    int fixMin = -1, fixMax = -1;
    if(gFixSM.count(hStream)) {
        auto p = gFixSM[hStream];
        fixMin = p.first; fixMax = p.second;
    }
    int agents = launchInfo->occup * nrSM; // NOT min(occup * nrSM, gridSize)!!!!
    dim3 gridDim(gridDimX, gridDimY, gridDimZ);
    nargs[numArgs - 6] = &gridDim;
    nargs[numArgs - 5] = &launchInfo->d_agents.d;
    nargs[numArgs - 4] = &launchInfo->fetched.d;
    nargs[numArgs - 3] = &launchInfo->finished.d;
    nargs[numArgs - 2] = &launchInfo->minSM.d;
    nargs[numArgs - 1] = &launchInfo->maxSM.d;
    // printf("Time to launch kernel %s: %lfus\n", launchInfo->name, getNano(hrc::now() - launchInfo->tLaunch) / 1000.0);
    // printf("[smsched] kernel launched: %s, func: %p, grid: (%d, %d, %d), block: (%d, %d, %d), sharedMemBytes: %u, stream: %p, numArgs: %d, occup: %d, agents: %d\n",
    //     launchInfo->name, (void*)f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, numArgs, launchInfo->occup, agents);
    // fflush(stdout);
    std::thread* profile = nullptr;
    std::vector<int> profileWaves;
    bool doProfile = with_profile && !gCI.count(launchInfo->name);
    if(doProfile) {
        generateProfileWaves(nrSM, launchInfo, profileWaves, fixMin, fixMax);
        launchInfo->h_minSM = 0;
        launchInfo->h_maxSM = getSMFromWave(profileWaves[0], launchInfo->occup);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    } else {
        launchInfo->h_minSM = 0;
        launchInfo->h_maxSM = nrSM;
    }
    gGdrPool->launch_init(launchInfo->slot, pack4(agents, 0), pack4(launchInfo->h_minSM, launchInfo->h_maxSM), true, hStream);
    if(doProfile) {
        profile = new std::thread(profileThread, launchInfo, std::ref(profileWaves));
    }
    CUresult res = CUDA_SUCCESS;
    // if(1
    //     && strstr(launchInfo->name, "NSL_IfLi4ELb1EEESF_EENS1R_25Til") == nullptr
    //     // && strstr(launchInfo->name, "redLoadIteratorMixedINS1S_1") == nullptr
    //     && strstr(launchInfo->name, "BatchDecode") == nullptr
    // )
    {
        // printf("[smsched] kernel launched: %s, func: %p, grid: (%d, %d, %d), block: (%d, %d, %d), sharedMemBytes: %u, stream: %p, numArgs: %d, occup: %d, agents: %d\n",
        //     launchInfo->name, (void*)f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, numArgs, launchInfo->occup, agents);
        res = real_cuLaunchKernel(f, agents, 1, 1, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, nargs, extra);
    }
    if(!profile) {
        gManagedLaunches.fetch_add(1, std::memory_order_relaxed);
        if(kstats) kstats->managed.fetch_add(1, std::memory_order_relaxed);
        if(gCI.count(launchInfo->name)) launchInfo->ci = &gCI[launchInfo->name];
    } else if(kstats) {
        kstats->profiled.fetch_add(1, std::memory_order_relaxed);
    }

    if(profile) {
        profile->join();
        delete profile;
        gGdrPool->launch_free(launchInfo->slot);
        delete launchInfo;
        launchInfo = nullptr;
    } else {
        streamInfo->pending->push(launchInfo);
        if(usePostLaunchSpecialMode() && hasPattern(launchInfo->name, "redLoadIteratorMixedINS1S_1")) {
            writeSM(launchInfo, 0, split_hint);
            launchInfo->isScheduled = 1;
        }
        if(usePostLaunchSpecialMode() && hasPattern(launchInfo->name, "NSL_IfLi4ELb1EEESF_EENS1R_25Til")) {
            writeSM(launchInfo, split_hint, nrSM);
            launchInfo->isScheduled = 1;
        }
        if(!disableLaunchTimeCILimitMode() && !launchInfo->isScheduled && isCI(launchInfo)) {
            // printf("[smsched] Adjusting CI kernel in stream %p\n", launchInfo->stream);
            writeSM(launchInfo, 0, split_hint);
            launchInfo->isScheduled = 1;
            gCILimitedLaunches.fetch_add(1, std::memory_order_relaxed);
            if(kstats) kstats->ciLimited.fetch_add(1, std::memory_order_relaxed);
        }
        // if(strstr(launchInfo->name, "redLoadIteratorMixedINS1S_1")) {
        //     writeSM(launchInfo, 0, split_hint);
        //     launchInfo->isScheduled = 1;
        // }
        // if(strstr(launchInfo->name, "NSL_IfLi4ELb1EEESF_EENS1R_25Til")) {
        //     writeSM(launchInfo, 76, nrSM);
        // }
    }

    for(int i = 0; i < numArgs - 6; ++i) {
        free(nargs[i]);
    }
    delete[] nargs;
    // fprintf(stderr, "[smsched] kernel finished\n");
    CU_CHECK(res);
    launchMutex.unlock();
    return res;
}

CUresult cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {
    if (shared_lib == NULL) {
        // fprintf(event_log, "[mod] cuLaunchKernelEx called before ld_init, calling ld_init now...\n");
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
        // fprintf(event_log, "[info] cuModuleLoad called before ld_init, calling ld_init now...\n");
        ld_init();
    }

    CUresult result = real_cuModuleLoad(module, fname); // call the symbol
    // fprintf(event_log, "[info] cuModuleLoad %d\n", result);
    return result;
}

#define cuLaunchKernel_ptsz cuLaunchKernel_ptsz_unmodified
#define cuLaunchKernelEx_ptsz cuLaunchKernelEx_ptsz_unmodified
#include "unmodified.c" // include the auto-generated code
#undef cuLaunchKernel_ptsz
#undef cuLaunchKernelEx_ptsz

CUresult cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus) {
    if (shared_lib == NULL) {
        // fprintf(event_log, "[mod] cuGetProcAddress called before ld_init, calling ld_init now...\n");
        ld_init();
    }
    // fprintf(stderr, "[mod] cuGetProcAddress_v2 called, symbol=%s, cudaVersion=%d, flags=%lu\n", symbol, cudaVersion, flags);
    // fflush(stderr);
    std::string sym_str(symbol);
    if (sym_str == "cuGetProcAddress") {
        *pfn = (void*)cuGetProcAddress_v2;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if (sym_str == "cuLaunchKernel") {
        *pfn = (void*)cuLaunchKernel;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if (sym_str == "cuLaunchKernel_ptsz") {
        *pfn = (void*)cuLaunchKernel_ptsz;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if(sym_str == "cuModuleLoadData") {
        *pfn = (void*)cuModuleLoadData;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if(sym_str == "cuLibraryLoadData") {
        *pfn = (void*)cuLibraryLoadData;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if(sym_str == "cuModuleLoadDataEx") {
        *pfn = (void*)cuModuleLoadDataEx;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if(sym_str == "cuModuleGetFunction") {
        *pfn = (void*)cuModuleGetFunction;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if(sym_str == "cuKernelGetFunction") {
        *pfn = (void*)cuKernelGetFunction;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if(sym_str == "cuLibraryGetKernel") {
        *pfn = (void*)cuLibraryGetKernel;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if(sym_str == "cuLibraryGetModule") {
        *pfn = (void*)cuLibraryGetModule;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if(sym_str == "cuModuleLoadFatBinary") {
        *pfn = (void*)cuModuleLoadFatBinary;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if(sym_str == "cuLaunchKernelEx") {
        *pfn = (void*)cuLaunchKernelEx;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if(sym_str == "cuLaunchKernelEx_ptsz") {
        *pfn = (void*)cuLaunchKernelEx_ptsz;
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
        // fprintf(event_log, "[mod] cuGetProcAddress called before ld_init, calling ld_init now...\n");
        ld_init();
    }
    // fprintf(stderr, "[mod] cuGetProcAddress called, symbol=%s, cudaVersion=%d, flags=%lu\n", symbol, cudaVersion, flags);
    // fflush(stderr);
    std::string sym_str(symbol);
    if (sym_str == "cuGetProcAddress") {
        *pfn = (void*)cuGetProcAddress;
        return CUDA_SUCCESS;
    } else if (sym_str == "cuLaunchKernel") {
        *pfn = (void*)cuLaunchKernel;
        return CUDA_SUCCESS;
    } else if (sym_str == "cuLaunchKernel_ptsz") {
        *pfn = (void*)cuLaunchKernel_ptsz;
        return CUDA_SUCCESS;
    } else if(sym_str == "cuModuleLoadData") {
        *pfn = (void*)cuModuleLoadData;
        return CUDA_SUCCESS;
    } else if(sym_str == "cuLibraryLoadData") {
        *pfn = (void*)cuLibraryLoadData;
        return CUDA_SUCCESS;
    } else if(sym_str == "cuModuleLoadDataEx") {
        *pfn = (void*)cuModuleLoadDataEx;
        return CUDA_SUCCESS;
    } else if(sym_str == "cuModuleGetFunction") {
        *pfn = (void*)cuModuleGetFunction;
        return CUDA_SUCCESS;
    } else if(sym_str == "cuKernelGetFunction") {
        *pfn = (void*)cuKernelGetFunction;
        return CUDA_SUCCESS;
    } else if(sym_str == "cuLibraryGetKernel") {
        *pfn = (void*)cuLibraryGetKernel;
        return CUDA_SUCCESS;
    } else if(sym_str == "cuLibraryGetModule") {
        *pfn = (void*)cuLibraryGetModule;
        return CUDA_SUCCESS;
    } else if(sym_str == "cuModuleLoadFatBinary") {
        *pfn = (void*)cuModuleLoadFatBinary;
        return CUDA_SUCCESS;
    } else if(sym_str == "cuLaunchKernelEx") {
        *pfn = (void*)cuLaunchKernelEx;
        return CUDA_SUCCESS;
    } else if(sym_str == "cuLaunchKernelEx_ptsz") {
        *pfn = (void*)cuLaunchKernelEx_ptsz;
        return CUDA_SUCCESS;
    }
    CUresult result = real_cuGetProcAddress(symbol, pfn, cudaVersion, flags, symbolStatus);
    return result;
}

#ifdef __cplusplus
}
#endif
