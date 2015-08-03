#include <ast.h>

Slice rev_permut(Slice s);

Tensor *run_seq(Ast *a, int nthreads, MemSpace *mem, SMap *named) {
    printf("Not implemented.\n");
    return NULL;
}

/*
// Delta-reductions for primitive operations.
Tensor *reduce_scale(struct Scale *s, Tensor *t) {
    uniq_tens(t);
    SCAL(t->len, s->fac, t->b->x, 1);
    return t;
}

*/

/*
Ast *reduce_transpose(struct Prim *p) {
    r = exec_tensor(a->t->a);
    if(a->n != a->t->n) {
        fprintf(stderr, "Number of indices doesn't match transpose.\n");
        return NULL;
    }
    return mkLit(do_transpose(r, a->t->perm));
}*/

/*
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
*/

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

/**************** Ast linking and refcounting routines. *************/
struct Node *node_ctor(Ast *a, int nref) {
    struct Node *n = malloc(sizeof(struct Node));
    n->op = a;
    n->val = NULL;
    n->nref = nref;
    n->visited = 0;
    return n;
}

// This always returns a Node, whose value is guaranteed not to be a ref.
static struct Node *zip_ast_rec(Ast *a, SMap *defs, Map *out) {
    Ast **t;
    int i, m;
    struct Node *n = map_get(out, &a);

    if(n == (struct Node *)out) {
        printf("Error! input graph contains a cycle at elem %p\n", a);
        return NULL;
    } else if(n != NULL) { // ref. already added.
        n->nref++;
        return n;
    }

    map_put(out, &a, out); // insert bad ref to detect cycles

    if(a->type == TRef) {
        Ast *b = smap_get(defs, a->ref);
        if(b == NULL) {
            printf("Error! Undefined variable, %s\n", a->ref);
            return NULL;
        }
        if( (n = zip_ast_rec(b, defs, out)) == NULL) {
            return NULL;
        }
        map_put(out, &a, n);
        return n;
    }
    m = ast_children(a, &t);
    for(i=0; i<m; i++) { // replace child ptrs
        if(zip_ast_rec(t[i], defs, out) == NULL) {
            return NULL;
        }
    }

    n = node_ctor(a, 1);
    map_put(out, &a, n);
    return n;
}

// TODO: Improve error reporting by setting problematic node on return...
Map *zip_ast(Ast *a, struct Node **n, SMap *named) {
    struct Node *r;
    Map *out = map_ctor(256, sizeof(void *));

    if( (r = zip_ast_rec(a, named, out)) == NULL) {
        map_dtor(&out);
        return NULL;
    }
    *n = r;

    return out;
}


Map *zip_ast_n(int n, Ast **a, struct Node **node, SMap *named) {
    struct Node *r;                                           
    int i;
    Map *out = map_ctor(256, sizeof(void *));

    for(i=0; i<n; i++) {
      if( (r = zip_ast_rec(a[i], named, out)) == NULL) {
        map_dtor(&out);
        return NULL;
      }
      node[i] = r;
    }
    
    return out;
}
