#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tdot.h"

/**** Visualization ****/
void fill_float(float *x, int n) {
    int i;
    for(i=0; i<n; i++)
        x[i] = (float)i;
}

void show_vec(float *x, int n) {
    int i;
    if(n == 0) { printf("\n"); return; }
    for(i=0; i<n-1; i++)
        printf("%f ", x[i]);
    printf("%f\n", x[i]);
}
void show_mat(float *x, int n, int m) {
    int i;
    for(i=0; i<n; i++,x+=m)
        show_vec(x, m);
}
void show_ivec(int *x, int n) {
    int i;
    if(n == 0) { printf("()\n"); return; }
    printf("(");
    for(i=0; i<n-1; i++)
        printf("%d, ", x[i]);
    printf("%d)", x[i]);
}

// Amway!
void show_plan(struct DotInfo *p) {
    int nc = p->nc;

    show_ivec(p->Cstride, nc);
    printf(" <- %f * ", p->alpha); show_ivec(p->Astride, nc+p->n);
    printf(" . "); show_ivec(p->Bstride, nc+p->n);
    printf(" via "); show_ivec(p->stride, p->n);

    printf("\nscontr = ");
    show_ivec(p->scontr, p->n);
    printf("\nsizes %d <- %d . %d", p->clen, p->alen, p->blen);
    printf("\nbeta = %f.\n", p->beta);
}


/**** Sub-calculations ****/
// also safe if stride = shape
static int get_strides(int *stride, const int *shape, int n) {
    int i, j, k = 1;
    for(i=n-1; i>=0; i--) {
        j = shape[i];
        stride[i] = k;
        k *= j;
    }
    return k;
}

struct DotInfo *calc_plan(float alpha, int na, const int *sa, const uint8_t *pa,
                                       int nb, const int *sb, const uint8_t *pb,
                          float beta,  int nc) {
    int n = (na+nb - nc)/2;
    int sp = 3*nc + 4*n;
    int Astride[20];
    int Bstride[20];
    struct DotInfo *info;
    int i, k;

    if(na > 20 || nb > 20) {
        return NULL;
    }
    info = (struct DotInfo *)malloc(sizeof(struct DotInfo) + sp*sizeof(int));
    info->len = sizeof(struct DotInfo) + sp*sizeof(int);

    info->alpha = alpha;
    info->beta = beta;
    info->nc = nc;
    info->n = n;

    LINK_INFO(info, n, nc);

    for(i=0; i<nc+n; i++) {
        info->Astride[i] = info->Bstride[i] = 0;
        info->Astride[i] = info->Bstride[i] = 0;
        info->Cstride[i] = -1;
    }
    for(i=0; i<na; i++) {
        if(pa[i] < 0 || pa[i] > n+nc) {
            printf("Error! Index %d of A maps to %d (out of bounds)!\n",
                    i, pa[i]);
            free(info); return NULL;
        }
        if(info->Cstride[pa[i]] != -1) {
            printf("Error! Duplicate output dimension %d (from A)!\n",
                    pa[i]);
            free(info); return NULL;
        }
        info->Cstride[pa[i]] = sa[i];
    }
    for(i=0; i<nb; i++) { // copy scontr from B-shape
        if(pb[i] < 0 || pb[i] > n+nc) {
            printf("Error! Index %d of B maps to %d (out of bounds)!\n",
                    i, pb[i]);
            free(info); return NULL;
        }
        if(pb[i] < nc) {
            if(info->Cstride[pb[i]] != -1) {
                // Allow direct products. (note this doesn't catch dupes from B)
                //printf("Error! Duplicate output dimension %d (from B)!\n",
                //        pb[i]);
                //free(info); return NULL;
            }
            info->Cstride[pb[i]] = sb[i];
        } else if(info->Cstride[pb[i]] != sb[i]) {
            printf("Error! A and B have different size for contraction dimension %d (%d vs %d)!\n",
                        pb[i], info->Cstride[pb[i]], sb[i]);
            free(info); return NULL;
        }
    }

    info->alen = get_strides(Astride, sa, na);
    info->blen = get_strides(Bstride, sb, nb);
    info->clen = get_strides(info->Cstride, info->Cstride, nc);
    for(i=0; i<na; i++)
        info->Astride[pa[i]] = Astride[i];
    for(i=0; i<nb; i++)
        info->Bstride[pb[i]] = Bstride[i];

    memcpy(info->scontr, info->stride, sizeof(int)*n);
    get_strides(info->stride, info->scontr, n);

    return info;
}

