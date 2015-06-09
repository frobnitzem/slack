
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}
