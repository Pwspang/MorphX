// Selects the parse.py output generated from the matching cuda.h
// regenerate with: python parse.py <cuda.h> <libcuda.so>
#if CUDA_VERSION >= 13000
#include "signature_cuda13.c"
#else
#include "signature_cuda12.c"
#endif
