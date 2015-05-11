#include <ast.h>

Slice rev_permut(Slice s);

// Interpret the Tensor Contraction Ast
//
// The memory management strategy of the interpreter deserves some comment.
// Hand-written code usually uses a minimum number of memory
// areas, and sends intermediate results to them, as in a circular
// buffer.  We can accomplish the same thing here by maintaining
// a list of active memory spaces during interpretation.
//
// The resulting exec function has type:
//
// Ast_Tensor -> ST MemSpace Ast_Lit_Tensor
//
// The binding expressions are refcounted.

int exec_tensor(Ast *a, SMap *env, Slice *m, Tensor *res) {
    switch(a->type) {
    };
    return 0;
}

// Traverse the DAG and give each object encountered a refcount.
// The initial refcount is assumed to be zero.
//static int count_refs() {
//    return 0;
//}

// Delta-reductions for primitive operations.
Tensor *reduce_scale(struct Scale *s, Tensor *t) {
    uniq_tens(t);
    SCAL(t->len, s->fac, t->b->x, 1);
    return t;
}

/*
Ast *reduce_transpose(struct Prim *p) {
    r = exec_tensor(a->t->a);
    if(a->n != a->t->n) {
        fprintf(stderr, "Number of indices doesn't match transpose.\n");
        return NULL;
    }
    return mkLit(do_transpose(r, a->t->perm));
}*/

Ast *reduce_sum(struct Prim *p, Tensor *a, Tensor *b) {
    if(a->b->nref == 1) { // accum. into a.
        uniq_tens(a);
        tsum(a, b, p->sum->perm);
        tensor_dtor(&b);
        return a;
    } else { // accum. into b
        uniq_tens(b);
        Slice rperm = rev_permut(p->sum->perm);
        tsum(b, a, rperm);
        slice_dtor(rperm);
        tensor_dtor(&a);
        return b;
    }
    return;
}

Tensor *reduce_dot(struct Prim *p, Tensor *a, Tensor *b, Tensor *c) {
    if(c->n != p->dot->n) {
        printf("Error - number of output dimensions doesn't match.\n");
        return NULL;
    }
    uniq_tens(c);
    tdot(a, p->dot->pa, b, p->dot->pb, c, c->n);
    tensor_dtor(&a);
    tensor_dtor(&b);
    return c;
}

// construct the inverse of a permutation
Slice rev_permut(Slice s) {
    Slice r = slice_ctor(s->width, s->n, s->n);
    int *x = s->x;
    int *y = r->x;
    int i;

    for(i=0; i<s->n; i++) {
        y[x[i]] = i;
    }
    return r;
}

