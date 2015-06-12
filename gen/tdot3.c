/*  Unfinished version to check for simplifications or
 * re-writes as series of GEMM-type operations.
 *
 */
#include <magma.h>
#include "tdot.h"

double wtime( void ) {
    struct timeval t;
    gettimeofday( &t, NULL );
    return t.tv_sec + t.tv_usec*1e-6;
}

double sync_wtime( cudaStream_t queue ) {
    gpuErrchk(cudaStreamSynchronize(queue));
    return wtime();
}

void tdot(struct DotInfo *info, const float *A, const float *B, float *C) {
    int la, lb;
    // Determine lowest dim index of A
    for(la=0; la<info->nc+info->n; la++)
        if(info->Astride[la] == 1)
            break;
    // Determine lowest dim index of B
    for(lb=0; lb<info->nc+info->n; lb++)
        if(info->Astride[lb] == 1)
            break;

    if(la < info->nc) {
        if(lb < info->nc) { // both outer indices
            if(info->n > 0) {
                sum_NT();
            }
            // Accumulate GER
            sum_ger(info, la, lb, A, B, C);
        } else { // A outer, B inner
            sum_NN(info, la, lb, A, B, C);
        }
    } else {
        if(lb < info->nc) { // A inner, B outer
            sum_TT();
        } else { // both inner indices
            // ck. if same or different contraction ind.
            sum_TN();
        }
    }
}

// Wrapper for last-minute planners.
void tensdot(float beta,  float *C, int nc,
             float alpha, float *A, int na, int *sa, uint8_t *pa,
                          float *B, int nb, int *sb, uint8_t *pb) {
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


