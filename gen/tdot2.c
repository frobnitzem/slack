#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "tdot.h"
#include "cuda_bits.h"

// Kernel API
void tdot1(struct DotInfo *info, const float *A, const float *B, float *C,
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
void tdot(struct DotInfo *info, float *A, float *B, float *C) {
    float *d_A, *d_B, *d_C;
    double time;
    cudaStream_t stream;

    gpuErrchk(cudaStreamCreate(&stream)); // O(250 ms)

    gpuErrchk(cudaMalloc((void **)&d_A, info->alen*sizeof(float)));
    gpuErrchk(cudaMemcpyAsync(d_A, A, info->alen*sizeof(float),
                    cudaMemcpyHostToDevice, stream));

    gpuErrchk(cudaMalloc((void **)&d_B, info->blen*sizeof(float)));
    gpuErrchk(cudaMemcpyAsync(d_B, B, info->blen*sizeof(float),
                    cudaMemcpyHostToDevice, stream));

    gpuErrchk(cudaMalloc((void **)&d_C, info->clen*sizeof(float)));
    gpuErrchk(cudaMemcpyAsync(d_C, C, info->clen*sizeof(float),
                    cudaMemcpyHostToDevice, stream));

    time = sync_wtime(stream);
    tdot1(info, d_A, d_B, d_C, stream);
    time = sync_wtime(stream) - time;

    gpuErrchk(cudaMemcpyAsync(C, d_C, info->clen*sizeof(float),
                    cudaMemcpyDeviceToHost, stream));

    gpuErrchk(cudaStreamSynchronize(stream));

    gpuErrchk(cudaFree(d_A));
    gpuErrchk(cudaFree(d_B));
    gpuErrchk(cudaFree(d_C));
    gpuErrchk(cudaStreamDestroy(stream));

    int inner = info->n > 0 ? info->scontr[0]*info->stride[0] : 1;
    printf("TDOT: %f Gflops\n", 2e-9*info->clen*inner/time);
}

#define N 16

int main(void) {
    float *A, *B, *C;
    /*int na = 3; int pa[] = {1,2,3}; int sa[] = {128,128,128};
    int nb = 2; int pb[] = {0,3};   int sb[] = {128,128};
    int nc = 3;                     int sc[] = {128,128,128};*/
    // tdot8_12_3T2_3A1_2B2_0
    /*int na = 2; uint8_t pa[] = {0,2}; int sa[] = {32,32};
    int nb = 2; uint8_t pb[] = {1,2}; int sb[] = {32,32};
    int nc = 2;                   int sc[] = {32,32};*/
    /*int na = 1; uint8_t pa[] = {1}; int sa[] = {4};
    int nb = 2; uint8_t pb[] = {1,0}; int sb[] = {4,3};
    int nc = 1; int sc[] = {3};*/
    // tdot32_32_1_1_16T4_4_1_1A0_2_4B1_3_4
    int na = 4; uint8_t pa[] = {4,2,0,5}; int sa[] = {8,1,N,8};
    int nb = 4; uint8_t pb[] = {1,5,4,3}; int sb[] = {1,8,8,N};
    int nc = 4;                           int sc[] = {N,1,1,N};

    struct DotInfo *info = calc_plan(1.0, na, sa, pa,
                                          nb, sb, pb,
                                     0.0, nc);
    if(info == NULL) {
        return 1;
    }

    show_plan(info);
    A  = (float *)malloc(info->alen*sizeof(float)); fill_float(A, info->alen);
    B  = (float *)malloc(info->blen*sizeof(float)); fill_float(B, info->blen);
    C  = (float *)malloc(info->clen*sizeof(float));
    memset(C, 0, info->clen*sizeof(float));

    //show_vec(A, sa[0]);
    //show_mat(B, sb[0], sb[1]);
    tdot(info, A, B, C);

    show_mat(C, sc[0], sc[1]);
    //show_vec(C, sc[0]);
    free(A); free(B); free(C);
    free(info);

    return 0;
}

// Wrapper for last-minute planners.
void tensdot(float alpha, float *A, int na, int *sa, uint8_t *pa,
                          float *B, int nb, int *sb, uint8_t *pb,
             float beta,  float *C, int nc) {
    struct DotInfo *info = calc_plan(alpha, na, sa, pa,
                                            nb, sb, pb,
                                     beta,  nc);
    if(info == NULL) { // error
        return;
    }

    show_plan(info);
    tdot(info, A, B, C);
    free(info);
}
