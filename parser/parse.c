#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include "parser.h"
#include "parser.tab.h"
#include "lexer.h"

//int tce2_debug = 1;

static uint8_t find_index(uint8_t u, Slice a);

SMap *tce2_parse_inp(struct Environ *e, FILE *f) {
    struct Lexer_Context ctxt = {
        .file = f,
        .esc_depth = 0,
        .nest = 0,
        .e = e,
        .ind_map = smap_ctor(32),
        .nindices = 0,
        .ret = smap_ctor(16),
    };

    tce2_lex_ctor(&ctxt);

    tce2_parse(&ctxt); // nonzero on parse error

    tce2_lex_dtor(&ctxt);
    smap_dtor(&ctxt.ind_map);
    printf("Total: %d indices.\n", ctxt.nindices);

    return ctxt.ret;
}

struct active *mkActive(Ast *a, Slice ind) {
    struct active *r = malloc(sizeof(struct active));
    r->a = a;
    r->scale = 1.0;
    r->ind = ind;
    return r;
}

void act_dtor(struct active *act) {
    slice_dtor(&act->ind);
    free(act);
}

// If necessary, form a transposition operation to get from act->a to output
// order ind.
Ast *ck_transpose(struct active *act, Slice ind) {
    int ord;
    uint8_t *perm;
    Ast *r;

    if(act->ind->n != ind->n) {
        fprintf(stderr, "Error! number of indices does not match!\n");
        exit(1); // return value not yet tested.
        return NULL;
    }

    if( (perm = get_perm(act->ind, ind, &ord)) == NULL) {
        exit(1); // return value not yet tested.
    }

    if(!ord) {
        r = mkTranspose(act->scale, act->a, ind->n, perm);
    } else if(act->scale != 1.0) {
        r = mkScale(act->scale, act->a);
    } else {
        r = act->a;
    }

    free(perm);
    return r;
}

uint8_t *get_perm(Slice ind, Slice out, int *is_ord) {
    int i, ord=1;
    uint8_t *dim = out->x;
    uint8_t *perm = malloc(out->n);

    for(i=0; i<out->n; i++) {
        if( (perm[i] = find_index(dim[i], ind)) == 255) {
            fprintf(stderr, "Error! index %d not "
                            "found in input tensor.\n", dim[i]);
            free(perm);
            return NULL;
        }
        ord &= (perm[i] == i);
    }
    *is_ord = ord;

    return perm;
}


// Index stitching during parsing.
// ca, cb are input index codes
// cc are the output index codes
// csum are the codes for the summation (contraction) indices
// ctr is a (contractions x 2) list of (0-based) positions
//   of the contracted indices
int partition_inds(Slice *cc_p, Slice *ctr_p, Slice csum, Slice ca, Slice cb) {
    uint8_t *cc, *ind;
    uint8_t *ctr;
    int nleft = ca->n + cb->n - 2*csum->n;
    int i, j, k;

    // Check for double-contraction of the same index.
    if(ck_duplicate(csum)) {
        fprintf(stderr, "Contracted index is listed twice!\n");
        return -1;
    }

    // Allocate output slices.
    (*cc_p) = slice_ctor(1, nleft, nleft);
    cc = (*cc_p)->x;
    (*ctr_p) = slice_ctor(1, 2*csum->n, 2*csum->n);
    ctr = (*ctr_p)->x;

    // Create ctr by traversing csum
    ind = csum->x;
    for(i=0; i<csum->n; i++) {
        if( (ctr[0] = find_index(ind[i], ca)) == 255) {
            fprintf(stderr, "Contraction index %d not found in A.\n", ind[i]);
            return -1;
        }
        if( (ctr[1] = find_index(ind[i], cb)) == 255) {
            fprintf(stderr, "Contraction index %d not found in B.\n", ind[i]);
            return -1;
        }
        ctr += 2;
    }

    // Create cc by traversing ca, then cb
    j = 0;
    ind = ca->x;
    for(i=0; i<ca->n; i++) {
        if(find_index(ind[i], csum) == 255) {
            cc[j++] = ind[i];
        }
    }
    ind = cb->x;
    for(i=0; i<cb->n; i++) {
        if(find_index(ind[i], csum) == 255) {
            cc[j++] = ind[i];
            for(k=0; k<ca->n - csum->n; k++) { // Check overlap.
                if(cc[k] == ind[i]) {
                    fprintf(stderr, "Output index %d exists on both "
                                    "left and right side.\n", ind[i]);
                    return -1;
                }
            }
        }
    }
    /*if(j != nleft) { // shouldn't be possible.
        fprintf(stderr, "Too few contraction indices!\n");
        return -1;
    }*/

    return csum->n;
}

static uint8_t find_index(uint8_t u, Slice a) {
    uint8_t *x = a->x;
    int i;
    for(i=0; i<a->n; i++) {
        if(x[i] == u) return i;
    }
    return 255;
}

// Returns nonzero if the slice (of uint8_t) contains a duplicated number.
int ck_duplicate(Slice a) {
    uint8_t *ind = a->x;
    int i, j;

    for(i=0; i<a->n-1; i++) {
        for(j=i+1; j<a->n; j++) {
            if(ind[i] == ind[j]) {
                return 1;
            }
        }
    }
    return 0;
}
