/*
 *   To allow future tiling of matrix operations,
 * each tensor operation spawns 2 quark tasks.  The
 * first allocates space for the result, and the second
 * runs the operation with a fixed output location.
 *
 *   Since Ast objects are immutable, the graph reduction
 * operates by building a map from (Ast *)-s to (struct Node *)-s
 * during a first (check) pass over the Ast.  This pass also eliminates
 * TRef-s by linking directly to the first non-Ref child,
 * as well as setting the refcounts (number of references to
 * the output of each object), and detecting cycles.
 *
 *   The second pass runs the DAG via quark (using the visited flag).
 * As it is being executed, each Node will hold the temporary
 * output location for the operation.
 *
 *   The output is a tensor, pointing at the only remaining active
 * element of the MemSpace.
 *
 *   Note that if one node has 2/more of the same input,
 * and wants to output to one of them, the current algorithm
 * always copies.  This could be optimized by comparing locations
 * and refcounts...
 *
 *   Also note that the refcount is in two places -- a static total in Node-s
 * as well as the dynamic current nref in MemSpace.  The dynamic one
 * requires a mutex.  A useful improvement would be to modify Quark
 * to use the memory locations (and refcounts?) in Node info structures.
 */

#include <ast.h>
#include <quark.h>

// not in quark.h despite manual...
#define NODEP INPUT
#define MEM_SZ (sizeof(void *))

// Add a subexpression to the running queue.
static struct Node *add_q(Quark *q, Ast *a, MemSpace *mem, Map *vis);

Tensor *run_quark(Ast *a, int nthreads, MemSpace *mem, SMap *named) {
    Map *vis;
    Quark *q;
    struct Node *n;
    Tensor *ret;

    if( (vis = zip_ast(a, &n, named)) == NULL) {
        printf("Error in zip_ast.\n");
        return NULL;
    }

    q = QUARK_New(nthreads);
    add_q(q, n->op, mem, vis); // have to use n->op, in case a->type == TRef
    // to re-run, set all visited = 0, val = NULL
    //QUARK_Barrier(q);
    QUARK_Delete(q);

    ret = n->val; // extract location of final output tensor
    map_dtor(&vis);
    return ret;
}

// stores outputs in **names
int run_quark_n(int n, void **names, int nthreads, MemSpace *mem, SMap *named) {
    Map *vis;
    Quark *q;
    struct Node *node[n];
    int i;

    // Lookup Ast-s from names.
    for(i=0; i<n; i++) {
	char *x = names[i];
	if( (names[i] = smap_get(named, x)) == NULL) {
	    printf("Error: %s not defined.\n", x);
	    return 1;
	}
    }
    if( (vis = zip_ast_n(n, (Ast **)names, node, named)) == NULL) {
        printf("Error in zip_ast.\n");
        return 1;
    }

    q = QUARK_New(nthreads);
    for(i=0; i<n; i++) {
	add_q(q, node[i]->op, mem, vis); // have to use n->op, in case a->type == TRef
    }
    // to re-run, set all visited = 0, val = NULL
    //QUARK_Barrier(q);
    QUARK_Delete(q);

    for(i=0; i<n; i++) {
	names[i] = node[i]->val; // extract location of final output tensor
    }
    map_dtor(&vis);
    return 0;
}

// get output shape of tdot
int *get_dot_shape(struct Dot *dot, Tensor *a, Tensor *b) {
    int *sz = malloc(sizeof(int)*dot->nc);
    int i, j;
    for(i=0; i<a->n; i++) {
        j = dot->pa[i];
        if(j < dot->nc)
            sz[j] = a->shape[i];
    }
    for(i=0; i<b->n; i++) {
        j = dot->pb[i];
        if(j < dot->nc)
            sz[j] = b->shape[i];
    }
    return sz;
}
// get output shape of tadd
int *get_add_shape(struct Add *add, Tensor *b) {
    int *sz = malloc(sizeof(int)*add->n);
    int i;
    for(i=0; i<add->n; i++) {
        sz[add->pb[i]] = b->shape[i];
    }
    return sz;
}

void qalloc_tdot(Quark *q) {
    struct Node *n, *a, *b, *c;
    MemSpace *mem;
    //fprintf(stderr, "qalloc_tdot\n");
    // macro
    quark_unpack_args_5(q, mem, n, a, b, c);
    if(c->val == NULL) {
        int *sz = get_dot_shape(n->op->dot, a->val, b->val);
        n->val = newTensor(n->op->dot->nc, sz, n->nref, mem);
        free(sz);
        if(n->op->dot->beta != 0.0) {
            printf("Strange TDot operation adding to Zero with beta = %f\n",
                   n->op->dot->beta);
            n->op->dot->beta = 0.0;
        }
    } else {
        n->val = uniq_tens(mem, c->val, n->nref);
    }
}
void qrun_tdot(Quark *q) {
    struct Node *n, *a, *b;
    struct Dot *dot;
    MemSpace *mem;

    //fprintf(stderr, "qrun_tdot\n");
    quark_unpack_args_4(q, mem, n, a, b);
    dot = n->op->dot;

    tdot(dot->beta,  n->val,
         dot->alpha, a->val, dot->pa, \
                     b->val, dot->pb);
    tensor_dtor(&a->val, mem);
    tensor_dtor(&b->val, mem);
}

void qalloc_tadd(Quark *q) {
    struct Node *n, *a, *b;
    MemSpace *mem;
    //fprintf(stderr, "qalloc_tadd\n");

    // macro
    quark_unpack_args_4(q, mem, n, a, b);

    if(a->val == NULL) {
        int *sz = get_add_shape(n->op->add, b->val);
        n->val = newTensor(n->op->add->n, sz, n->nref, mem);
        free(sz);
        if(n->op->add->alpha != 0.0) {
            printf("Strange TAdd operation adding to Zero with alpha = %f\n",
                   n->op->add->alpha);
            n->op->add->alpha = 0.0;
        }
    } else {
        n->val = uniq_tens(mem, a->val, n->nref);
    }
}
void qrun_tadd(Quark *q) {
    struct Node *n, *b;
    struct Add *add;
    MemSpace *mem;

    //fprintf(stderr, "qrun_tadd\n");

    quark_unpack_args_3(q, mem, n, b);
    add = n->op->add;
    tadd(add->alpha, n->val,
         add->beta,  b->val, add->pb);
    tensor_dtor(&b->val, mem);
}

void qalloc_scale(Quark *q) {
    struct Node *n;
    struct Node *a;
    MemSpace *mem;
    quark_unpack_args_3(q, mem, n, a);
    n->val = uniq_tens(mem, a->val, n->nref);
}
void qrun_scale(Quark *q) {
    struct Node *n;
    MemSpace *mem;

    quark_unpack_args_2(q, mem, n);
    tscale(n->op->scale->alpha, n->val);
}

// min : Map string (Ast *)
// mout : Map (Ast *) (Ast *)
//
// Add via topo-sort traversal to ensure proper run-ordering.
//
// This will link all the refs in 'a' and set up the runtime
// to use a->val to link to their output.
//
//   If the Ast came with subexpression time estimates, priorities
// could also be added here.
//
// This runs single-threaded, so no contention on vis occurs.
static struct Node *add_q(Quark *q, Ast *op, MemSpace *mem, Map *vis) {
    Quark_Task_Flags tflags = Quark_Task_Flags_Initializer;
    struct Node *n = map_get(vis, &op);
    struct Node *a, *b, *c;

    if(n == NULL) {
        fprintf(stderr, "Error looking up node %p!\n", op);
        return NULL;
    }
    if(n->visited) {
        return n;
    }
    n->visited = 1;
    op = n->op; // Leap past Ref-s.

    switch(op->type) {
    case TDot:
        a = add_q(q, op->dot->a, mem, vis);
        b = add_q(q, op->dot->b, mem, vis);
        c = add_q(q, op->dot->c, mem, vis);
        QUARK_Insert_Task(q, qalloc_tdot, NULL,
                          MEM_SZ, mem, NODEP,
                          sizeof(struct Node), n, INOUT,
                          sizeof(struct Node), a, INPUT,
                          sizeof(struct Node), b, INPUT,
                          sizeof(struct Node), c, INPUT,
                          0);
        QUARK_Task_Flag_Set(&tflags, TASK_LABEL, (intptr_t)"TDot");
        QUARK_Insert_Task(q, qrun_tdot, &tflags,
                          MEM_SZ, mem, NODEP,
                          sizeof(struct Node), n, INOUT,
                          sizeof(struct Node), a, INPUT,
                          sizeof(struct Node), b, INPUT,
                          0); // actually all of the above, but alloc gets it
        break;
    case TAdd:
        a = add_q(q, op->add->a, mem, vis);
        b = add_q(q, op->add->b, mem, vis);
        QUARK_Insert_Task(q, qalloc_tadd, NULL,
                          MEM_SZ, mem, NODEP,
                          sizeof(struct Node), n, INOUT,
                          sizeof(struct Node), a, INPUT,
                          sizeof(struct Node), b, INPUT,
                          0);
        QUARK_Task_Flag_Set(&tflags, TASK_LABEL, (intptr_t)"TAdd");
        QUARK_Insert_Task(q, qrun_tadd, &tflags,
                          MEM_SZ, mem, NODEP,
                          sizeof(struct Node), n, INOUT,
                          sizeof(struct Node), b, INPUT,
                          0);
        break;
    case TScale:
        a = add_q(q, op->scale->a, mem, vis);
        QUARK_Insert_Task(q, qalloc_scale, NULL,
                          MEM_SZ, mem, NODEP,
                          sizeof(struct Node), n, INOUT,
                          sizeof(struct Node), a, INPUT,
                          0);
        QUARK_Task_Flag_Set(&tflags, TASK_LABEL, (intptr_t)"TScale");
        QUARK_Insert_Task(q, qrun_scale, &tflags,
                          MEM_SZ, mem, NODEP,
                          sizeof(struct Node), n, INOUT,
                          0);
        break;
    case TBase:
        if(op->base->type == BTens) {
            n->val = op->base->t; // set output location
                                 // tell allocator to ignore it
            insert_unmanaged(mem, n->val->x, sizeof(double)*n->val->len);
        } else if(op->base->type == BZeroTens) {
            n->val = NULL;
        } else {
            fprintf(stderr, "Error! Unknown / non-implemented Base type: %d\n",
                        op->base->type);
        }
        break;
    default:
        fprintf(stderr, "Error! Unknown / non-implemented Ast type: %d\n", op->type);
        // Note that this leaves n->val == NULL
        // and will likely cause a segfault later.
    }
    return n;
}
