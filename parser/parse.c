#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include "parser.h"
#include "parser.tab.h"
#include "lexer.h"

//int tce2_debug = 1;

static int find_index(uint8_t u, Slice a);

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
    int i, ord;
    uint8_t *dim = ind->x;
    uint8_t *perm;
    Ast *r;

    if(act->ind->n != ind->n) {
        fprintf(stderr, "Error! number of indices does not match!\n");
        exit(1); // return value not yet tested.
        return NULL;
    }

    perm = malloc(ind->n);
    ord = 1;
    for(i=0; i<ind->n; i++) {
        if( (perm[i] = find_index(dim[i], act->ind)) < 0) {
            fprintf(stderr, "Error! index %d not "
                            "found in input tensor.\n", dim[i]);
            exit(1); // return value not yet tested.
            return NULL;
        }
        ord &= (perm[i] == i);
    }

    if(ord) { // result is already ordered.
        free(perm);
        return act->a;
    }
    r = mkTranspose(act->a, ind->n, perm);
    free(perm);

    return r;
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
    int i, j;

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
        if( (ctr[0] = find_index(ind[i], ca)) < 0) {
            fprintf(stderr, "Contraction index %d not found in A.\n", ind[i]);
            return -1;
        }
        if( (ctr[1] = find_index(ind[i], cb)) < 0) {
            fprintf(stderr, "Contraction index %d not found in B.\n", ind[i]);
            return -1;
        }
        ctr += 2;
    }

    // Create cc by traversing ca, then cb
    j = 0;
    ind = ca->x;
    for(i=0; i<ca->n; i++) {
        if(find_index(ind[i], csum) < 0) {
            cc[j++] = ind[i];
        }
    }
    ind = cb->x;
    for(i=0; i<cb->n; i++) {
        if(find_index(ind[i], csum) < 0) {
            cc[j++] = ind[i];
        }
    }
    /*if(j != nleft) { // shouldn't be possible.
        fprintf(stderr, "Too few contraction indices!\n");
        return -1;
    }*/

    return csum->n;
}

static int find_index(uint8_t u, Slice a) {
    uint8_t *x = a->x;
    int i;
    for(i=0; i<a->n; i++) {
        if(x[i] == u) return i;
    }
    return -1;
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
