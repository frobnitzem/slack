#include <stdint.h>

// Cstride : Vec Int nc -- strides for C (scontr is logically just after)
// scontr : Vec Int n -- length of contracted dims
// stride : Vec Int n+1
//               -- contraction strides (including outer)
//             i.e. for (4,5,7), stride = (1,4,20,140)
// Astride : Vec Int nc+n -- perm(pa, indprod(Ashape)) [unset dims = 0]
// Bstride -- similar
struct DotInfo {
    size_t len; // len of this struct
    int nc, n;
    int alen, blen, clen;
    int *stride, *Astride, *Bstride, *scontr; // must be pointed into Cstride
    float alpha, beta;
    int Cstride[0]; // variable length
};

#define LINK_INFO(info, n, nc) { \
    info->stride  = info->Cstride + nc; \
    info->scontr  = info->Cstride + nc+n; \
    info->Astride = info->Cstride + nc+2*n; \
    info->Bstride = info->Cstride + nc+2*n + (nc+n); \
}

struct DotInfo *calc_plan(float beta,  int nc,
            float alpha, int na, const int *sa, const uint8_t *pa,
                         int nb, const int *sb, const uint8_t *pb);

void fill_float(float *x, int n);
void show_vec(float *x, int n);
void show_mat(float *x, int n, int m);
void show_ivec(int *x, int n);
void show_plan(struct DotInfo *p);

