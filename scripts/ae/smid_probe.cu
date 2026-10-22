// Records %smid / %nsmid for every block, then prints distinct SM IDs used.
#include <cstdio>
#include <cstdlib>
#include <set>
#include <unistd.h>
__global__ void probe(unsigned *smid, unsigned *nsmid, long long spin) {
    unsigned s, n;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    asm volatile("mov.u32 %0, %%nsmid;" : "=r"(n));
    long long t0 = clock64();
    while (clock64() - t0 < spin) {}
    if (threadIdx.x == 0) { smid[blockIdx.x] = s; nsmid[blockIdx.x] = n; }
}
int main(int argc, char **argv) {
    const char *tag = argc > 1 ? argv[1] : "probe";
    int nblk = 132 * 16;
    long long spin = argc > 2 ? atoll(argv[2]) : 1000000;
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    unsigned *d_s, *d_n, *h_s = new unsigned[nblk], *h_n = new unsigned[nblk];
    cudaMalloc(&d_s, nblk * 4); cudaMalloc(&d_n, nblk * 4);
    probe<<<nblk, 128>>>(d_s, d_n, spin);
    cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(h_s, d_s, nblk * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_n, d_n, nblk * 4, cudaMemcpyDeviceToHost);
    std::set<unsigned> ids(h_s, h_s + nblk);
    printf("[%s] pid=%d dev=%s cc=%d.%d multiProcessorCount=%d err=%s\n", tag, getpid(), p.name, p.major, p.minor,
           p.multiProcessorCount, cudaGetErrorString(e));
    printf("[%s] nsmid=%u distinct_smid=%zu min=%u max=%u\n", tag, h_n[0], ids.size(), *ids.begin(), *ids.rbegin());
    printf("[%s] ids:", tag); for (unsigned i : ids) printf(" %u", i); printf("\n");
    return e != cudaSuccess;
}
