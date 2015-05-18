#include <stdio.h>
#include <stdlib.h>
#include <tens.h>

// from tdot.c
void print_vec(int *x, int n);


void tdot(const double alpha, Tensor *a, const uint8_t *pa,
                              Tensor *b, const uint8_t *pb,
          const double  beta, Tensor *c) {
    tensdot(alpha, a->x, a->n, a->shape, pa,
                   b->x, b->n, b->shape, pb,
             beta, c->x, c->n);
}

void tadd(const double alpha, Tensor *a,
          const double beta,  Tensor *b, const uint8_t *pb) {
    int i;
    if(a->n != b->n) {
        goto err;
    }
    for(i=0; i<a->n; i++) {
        if(a->shape[i] != b->shape[pb[i]]) goto err;
    }
    tensadd(alpha, a->x, a->n, a->shape,
            beta,  b->x, pb);
    return;

err:
    printf("Error! Adding tensors with shapes: ");
    print_vec(a->shape, a->n); printf(" and "); print_vec(b->shape, b->n);
    printf("\n");
    for(i=0; i<a->n; i++)
        printf(" %u", pb[i]);
    printf(".\n");
}

void tscale(const double alpha, Tensor *a) {
    int i;
    int j=1;
    for(i=0; i<a->n; i++) {
        j *= a->shape[i];
    }
    if(j != a->len) {
        printf("Error! Tensor with shape ");
        print_vec(a->shape, a->n);
        printf(" has len = %d\n", a->len);
        return;
    }
    // i = 1;
    //dscal_(&j, &alpha, a->c, &i);

    for(i=0; i<a->len; i++) {
        a->x[i] *= alpha;
    }
}

/*  Memory management of tensor object is a little distraught,
 *  since tensors are a header attached to a block of data.
 *
 *  So, the convention here is to use the refcount on the
 *  block of data the same as that for the header.
 *
 *  Also, negative refcounts indicate unmanaged objects.
 *  A copy of an unmanaged object becomes a managed object.
 */

// Construct the tensor header (t->x = NULL);
Tensor *tensor_ctor(const int nd, const int *shape) {
    int i;
    Tensor *t = (Tensor *)malloc(sizeof(Tensor)+sizeof(int)*nd);

    t->len = 1;
    for(i=0; i<nd; i++)
        t->len *= shape[i];

    memcpy(t->shape, shape, sizeof(int)*nd);
    t->n = nd;
    t->x = NULL;

    return t;
}

// Note: this is called on unmanaged objects as well,
// where it is expected to do nothing (hence the deferral
// to release_block_if on whether to set t = 0).
void tensor_dtor(Tensor **t, MemSpace *mem) {
    if(mem != NULL) {
        release_block_if(mem, (*t)->x, (void **)t);
    } else {
        free(*t);
        *t = NULL;
    }
}

// Allocate a new tensor.
// x will be managed by MemSpace.
Tensor *mkTensor(const int nd, const int *shape,
                 const int nref, MemSpace *mem) {
    Tensor *t = tensor_ctor(nd, shape);
    t->x = (double *)reserve_block(mem, sizeof(double)*t->len, nref);
    return t;
}

// Cast a block of mem. to a tensor.
// x won't be managed by MemSpace, and the user is responsible
// for eventually free()-ing t.
Tensor *toTensor(double *x, const int nd, const int *shape, MemSpace *mem) {
    Tensor *t = tensor_ctor(nd, shape);
    t->x = x;
    insert_unmanaged(mem, x, sizeof(double)*t->len);
    return t;
}

// Allocate a tensor from managed mem.
// Note: this leaves the mem. uninitialized.
// If desired, use: memset(t->x, 0, sizeof(double)*t->len);
Tensor *newTensor(const int nd, const int *shape,
                  const int nref, MemSpace *mem) {
    Tensor *t = mkTensor(nd, shape, nref, mem);
    return t;
}

Tensor *uniq_tens(MemSpace *mem, Tensor *t, const int nref) {
    Tensor *u;
    void *y = uniq_block(mem, t->x, sizeof(double)*t->len, nref);
    if(y == t->x) { // already uniq
        return t;
    }
    u = tensor_ctor(t->n, t->shape);
    u->x = y;
    return u;
}

// Compose permutation perm after x, storing the result in x.
//
// The first, x, is moved forward according to the second, perm.
// Indices from x in the range [n, m) are left alone.
void compose_permut(const int m, uint8_t *x, const int n, const uint8_t *perm) {
    int i;
    for(i=0; i<m; i++) {
        if(x[i] < n)
            x[i] = perm[x[i]];
    }
}

// Construct the inverse of a permutation.
uint8_t *inv_permut(const int n, const uint8_t *x) {
    uint8_t *y = malloc(n);
    int i;
    for(i=0; i<n; i++) {
        y[x[i]] = i;
    }
    return y;
}

