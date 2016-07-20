/* This is template code for testing a given tdot kernel.
 * and requires filling in the necessary computations 
 * for sizes, etc. below.
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cuda_bits.h"

// template parameters:
#define KERN %(name)s
#define NC %(pass_nc)s
#define ASZ (%(comp_Asz)s)
#define BSZ (%(comp_Bsz)s)
#define CSZ (%(comp_Csz)s)

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

void KERN(%(cints)s,
          float alpha, const float* __restrict__ A,
                       const float* __restrict__ B,
          float  beta,       float* __restrict__ C,
          cudaStream_t stream);

typedef cudaStream_t cudaStream_t;

double wtime( void ) {
    struct timeval t;
    gettimeofday( &t, NULL );
    return t.tv_sec + t.tv_usec*1e-6;
}

double sync_wtime( cudaStream_t queue ) {
    gpuErrchk(cudaStreamSynchronize(queue));
    return wtime();
}

// Run a tdot on the cuda device.
// Note that the permutations are hard-wired inside the kernel.
// For this to work, the shapes have to be at least as large as
// the tile-sz.
void dyn_tdot(float beta,  float *C, int32_t *sc,
              float alpha, float *A, float *B) {
    float *d_A, *d_B, *d_C;
    double time, min;
    int asz = ASZ;
    int bsz = BSZ;
    int csz = CSZ;
    double size = sqrt((double)asz*bsz*csz); // number of mults
    int i;
    cudaStream_t stream;

    gpuErrchk(cudaStreamCreate(&stream)); // O(250 ms)

    gpuErrchk(cudaMalloc((void **)&d_A, asz*sizeof(float)));
    gpuErrchk(cudaMemcpyAsync(d_A, A, asz*sizeof(float),
                    cudaMemcpyHostToDevice, stream));

    gpuErrchk(cudaMalloc((void **)&d_B, bsz*sizeof(float)));
    gpuErrchk(cudaMemcpyAsync(d_B, B, bsz*sizeof(float),
                    cudaMemcpyHostToDevice, stream));

    gpuErrchk(cudaMalloc((void **)&d_C, csz*sizeof(float)));
    gpuErrchk(cudaMemcpyAsync(d_C, C, csz*sizeof(float),
                    cudaMemcpyHostToDevice, stream));

    for(i=0; i<10; i++) {
        time = sync_wtime(stream);
        KERN(NC, alpha, d_A, d_B, beta, d_C, stream);

        time = sync_wtime(stream) - time;
        if(i == 0 || time < min) min = time;
    } time = min;

    gpuErrchk(cudaMemcpyAsync(C, d_C, csz*sizeof(float),
                    cudaMemcpyDeviceToHost, stream));

    gpuErrchk(cudaStreamSynchronize(stream));

    gpuErrchk(cudaFree(d_A));
    gpuErrchk(cudaFree(d_B));
    gpuErrchk(cudaFree(d_C));
    gpuErrchk(cudaStreamDestroy(stream));

    printf(TOSTRING(KERN) ": %%f s => %%f Gflops\n", time, 2e-9*size/time);
}

