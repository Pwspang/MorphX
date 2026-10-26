// Two-stream check of the LithOS-style runtime (runtime/src/cuda-lithos.cpp).
// Build with clang++ and -fpass-plugin=libsmsched.so, run with the lithos hook
// preloaded and SMSCHED_SPLIT_HINT set. Stream A launches its first patched
// kernel before stream B. Each block records %smid; after both streams are
// registered, both launch kernels concurrently from two threads and every
// block must sit inside its stream's partition:
//   SMSCHED_LITHOS_RANK=registration: A in [0, split), B in [split, nrSM)
//   SMSCHED_LITHOS_RANK=priority with LITHOS_TEST_B_HIGH=1 (B created with the
//   greatest priority): B in [0, split), A in [split, nrSM)
// Phase 3 runs B alone; with SMSCHED_LITHOS_STEAL=1 it may use every SM, and
// phase 2 only checks results (slices may steal the other partition).
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <thread>
#include <vector>

__global__ void spin_mark(float *out, unsigned *smid, int n, long long spin) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    long long t0 = clock64();
    while(clock64() - t0 < spin) {}
    if(i < n) out[i] = 1.0f + blockIdx.x;
    if(threadIdx.x == 0) {
        unsigned s;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
        smid[blockIdx.x] = s;
    }
}

struct Run {
    cudaStream_t s;
    float *out;
    unsigned *smid;
    int blocks, threads;
};

static int nsm;

// Launch once and check results; returns the number of blocks outside [lo, hi).
static int launchCheck(Run &r, long long spin, int lo, int hi, const char *tag, int *wrong, std::set<unsigned> *ids) {
    int n = r.blocks * r.threads;
    cudaMemsetAsync(r.out, 0, n * sizeof(float), r.s);
    cudaMemsetAsync(r.smid, 0xff, r.blocks * sizeof(unsigned), r.s);
    spin_mark<<<r.blocks, r.threads, 0, r.s>>>(r.out, r.smid, n, spin);
    cudaError_t e = cudaStreamSynchronize(r.s);
    if(e != cudaSuccess) {
        printf("[%s] CUDA error %s\n", tag, cudaGetErrorString(e));
        exit(2);
    }
    std::vector<float> ho(n);
    std::vector<unsigned> hs(r.blocks);
    cudaMemcpy(ho.data(), r.out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hs.data(), r.smid, r.blocks * sizeof(unsigned), cudaMemcpyDeviceToHost);
    int outside = 0;
    for(int i = 0; i < n; ++i) *wrong += ho[i] != 1.0f + i / r.threads;
    for(unsigned v : hs) {
        if(v == 0xffffffffu) { ++*wrong; continue; }
        ids->insert(v);
        if((int)v < lo || (int)v >= hi) ++outside;
    }
    return outside;
}

int main() {
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    int split = getenv("SMSCHED_SPLIT_HINT") ? atoi(getenv("SMSCHED_SPLIT_HINT")) : nsm / 2;
    bool bHigh = getenv("LITHOS_TEST_B_HIGH") && atoi(getenv("LITHOS_TEST_B_HIGH"));
    const char *rank = getenv("SMSCHED_LITHOS_RANK");
    bool priority = rank && std::string(rank) == "priority";
    long long spin = getenv("LITHOS_TEST_SPIN") ? atoll(getenv("LITHOS_TEST_SPIN")) : 20000;
    int iters = getenv("LITHOS_TEST_ITERS") ? atoi(getenv("LITHOS_TEST_ITERS")) : 20;

    int least = 0, greatest = 0;
    cudaDeviceGetStreamPriorityRange(&least, &greatest);
    Run a{}, b{};
    a.blocks = b.blocks = nsm * 16;
    a.threads = b.threads = 128;
    cudaStreamCreateWithPriority(&a.s, cudaStreamNonBlocking, least);
    cudaStreamCreateWithPriority(&b.s, cudaStreamNonBlocking, bHigh ? greatest : least);
    for(Run *r : {&a, &b}) {
        cudaMalloc(&r->out, r->blocks * r->threads * sizeof(float));
        cudaMalloc(&r->smid, r->blocks * sizeof(unsigned));
    }

    int fails = 0, wrong = 0;
    std::set<unsigned> idsA, idsB;
    // Phase 1: register A, then B (also profiles the kernel once).
    launchCheck(a, spin, 0, nsm, "warmup A", &wrong, &idsA);
    launchCheck(b, spin, 0, nsm, "warmup B", &wrong, &idsB);
    printf("[warmup] wrong=%d\n", wrong);
    fails += wrong != 0;

    // Phase 2: both streams concurrently.
    bool aFirst = !(priority && bHigh);
    int aLo = aFirst ? 0 : split, aHi = aFirst ? split : nsm;
    int bLo = aFirst ? split : 0, bHi = aFirst ? nsm : split;
    int outA = 0, outB = 0, wrongA = 0, wrongB = 0;
    idsA.clear(); idsB.clear();
    std::thread ta([&] { for(int i = 0; i < iters; ++i) outA += launchCheck(a, spin, aLo, aHi, "A", &wrongA, &idsA); });
    std::thread tb([&] { for(int i = 0; i < iters; ++i) outB += launchCheck(b, spin, bLo, bHi, "B", &wrongB, &idsB); });
    ta.join(); tb.join();
    // With SMSCHED_LITHOS_STEAL=1 a slice may take the whole GPU whenever the
    // other stream has nothing in flight, so partitions are informational there.
    bool steal = getenv("SMSCHED_LITHOS_STEAL") && atoi(getenv("SMSCHED_LITHOS_STEAL"));
    bool ok = wrongA == 0 && wrongB == 0 && (steal || (outA == 0 && outB == 0));
    fails += !ok;
    printf("[concurrent] A expect [%d,%d) sms=%zu min=%d max=%d outside=%d wrong=%d | B expect [%d,%d) sms=%zu min=%d max=%d outside=%d wrong=%d -> %s\n",
        aLo, aHi, idsA.size(), idsA.empty() ? -1 : (int)*idsA.begin(), idsA.empty() ? -1 : (int)*idsA.rbegin(), outA, wrongA,
        bLo, bHi, idsB.size(), idsB.empty() ? -1 : (int)*idsB.begin(), idsB.empty() ? -1 : (int)*idsB.rbegin(), outB, wrongB,
        ok ? "PASS" : "FAIL");

    // Phase 3: B alone. Report how many SMs it used (informational).
    idsB.clear();
    int w3 = 0;
    for(int i = 0; i < 5; ++i) launchCheck(b, spin, 0, nsm, "B alone", &w3, &idsB);
    fails += w3 != 0;
    printf("[B alone] sms=%zu min=%d max=%d wrong=%d -> %s\n", idsB.size(),
        idsB.empty() ? -1 : (int)*idsB.begin(), idsB.empty() ? -1 : (int)*idsB.rbegin(), w3, w3 == 0 ? "PASS" : "FAIL");

    printf("RESULT: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
