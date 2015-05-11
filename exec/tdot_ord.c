// Routines for checking whether tensor contractions
// may be re-written as GEMM.
//
// For tensordot(A, B), uncontracted indices from A are on the left
// (most sig.) while those from B are all on the right (least sig.)
//
// The contraction indices have to be adjacent in memory and in the
// same order for both A and B, so these functions basically do a
// gold-plated GEMM.
//
// cnum-s are marker arrays, nonzero for contracted indices
static int *number_indices(int n, int nc, int *ind);
static int copy_shape(const Tensor *a, const int *cnum, int *shape);
static int ck_ordered(int n, const int *cnum);
static int ck_same(int nc, const int *ind);

// Tensors must be sorted from most to least significant dimension.
// nc is the number of contracted indices, and
// ind is an array of (nx by 2) integers (pairs of contracted indices).
Tensor *tensdot(Tensor *a, Tensor *b, int nc, int *ind, int nref, Slice *m) {
    int i = 0;
    int ord_a, ord_b;
    int *anum, *bnum;
    int an, bn;
    Tensor *t;

#ifdef PARANOID
    if(nc > a->n || nc > b->n) {
        fprintf(stderr, "tensdor: more contracted than input dims.\n");
        exit(1);
    }
#endif
    anum = number_indices(a->n, nc, ind);
    bnum = number_indices(b->n, nc, ind+1);

    t = tensor_ctor(a->n + b->n - 2*nc);
    an = copy_shape(a, anum, t->shape);
    bn = copy_shape(b, bnum, t->shape + a->n - nc);
    t->len = an*bn;
    t->b = reserve_block(m, nref, t->len*sizeof(float));

    if(nc == 0) { // trivial case A <- a X Y^T + A, A m x n
        GER(bn, an, 1.0, b->b->x, 1, a->b->x, 1,
                t->b->x, an);
        //GER(bn, an, a->scale*b->scale, b->b->x, 1, a->b->x, 1,
        //        t->b->x, an);
        free(anum); free(bnum);
        return t;
    }

    if( (ord_a = ck_ordered(a->n, anum))) {
        if( (ord_b = ck_ordered(b->n, bnum))
                && ck_same(nc, ind)) { // Straightforward dgemm.
            GEMM(ord_b == 1, ord_a == -1, bn, an, a->len / an,
                    1.0, a->b->x, an, b->b->x, bn,
                    0.0, t->b->x, bn);
            //GEMM(ord_b == 1, ord_a == -1, bn, an, a->len / an,
            //      a->scale*b->scale, a->b->x, an, b->b->x, bn,
            //      0.0, t->b->x, bn);
        // TODO: decide if transposing A would give better perf.
        } else { // Need to transpose B.
            printf("Need to transpose B.\n");
        }
    } else if( (ord_b = ck_ordered(b->n, bnum))) { // Need to transpose A
            printf("Need to transpose A.\n");
    } else { // Need to transpose both A and B.
            printf("Need to transpose A & B.\n");
    }

    free(anum);
    free(bnum);
    return t;
}

// Create a list of ints numbering the contractions (from 1,
// 0 => index not contracted).
static int *number_indices(int n, int nc, int *ind) {
    int *cnum;
    int i;

    cnum = calloc(n, sizeof(int));
    for(i=0; i<nc; i++) { // mark contracted indices
#ifdef PARANOID
        if(cnum[ind[2*i]]) {
            fprintf(stderr, "Duplicated contraction index.\n");
            exit(1);
        }
#endif
        cnum[ind[2*i]] = i+1;
    }

    return cnum;
}

// Copy the output shapes (to last arg)
static int copy_shape(const Tensor *a, const int *cnum, int *shape) {
    int i, n=0;
    int tot=1;

    for(i=0; i<a->n; i++) {
        if(!cnum[i]) {
            shape[n++] = a->shape[i];
            tot *= a->shape[i];
        }
    }
    return tot;
}

// Counts the number of contracted indices in the permutation, p.
// It also sets ord to indicate if p is matrix-like:
//   0 : mixed contracted / uncontracted indices (or out-of-order)
//  -1 : contracted indices are all on left (major)
//   1 : contracted indices are all on right (minor)
// Note that for actual matrix structures, there is always
// one contracted and one uncontracted -- always giving either 1 or -1.
static int ck_ordered(int n, const int *cnum) {
    int i;
    int state;


    if(n < 1) { // trivial cases -- let's say ordered
        return 1;
    }
    state = 1 + cnum[0];
    // state 0: unordered (interleaved)
    // state 1: all uncontracted indices
    // state 2: all contracted indices
    // state 3: uncontracted followed by contracted
    // state 4: contracted followed by uncontracted

    for(i=1; i<n; i++) {
        if(cnum[i]) { // contraction index (1 -> 3, 4 -> 0)
            if(state == 1) {
                state = 3;
            } else if(state == 4) {
                state = 0;
            }
        } else if(state == 2) { // uncontracted index (2 -> 4, 3 -> 0)
            state = 4;
        } else if(state == 3) {
            state = 0;
        }
    }

    switch(state) {
    case 0:
        return 0;
    case 1:
    case 2:
    case 3:
        return 1; // most favorable outcome.
    case 4:
        return -1;
    }
    return 0;
}

// Check that the contraction ordering is the same.
//
// Criteria for same ordering is that the contraction
// index for array 1 is the same as that for array 2, plus
// a constant offset.
// This routine segfaults if nc < 1
static int ck_same(int nc, const int *ind) {
    int off = ind[1]-ind[0];
    int i;
    const int *i2 = ind+2;

    for(i=1; i<nc; i++,ind+=2) { // check index within ordered
        if(i2[1]-i2[0] != off)
            return 0;
    }
    return 1;
}
