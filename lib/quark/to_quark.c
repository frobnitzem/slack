/*  To allow future tiling of matrix operations,
 * the graph is converted to an ST(MemSpace) operation.
 *
 *  Each tensor operation takes as input both its own Ast
 * (holding child pointers to input values), and
 * a 'void *' denoting the output location.
 * 
 *  In the future, Ast nodes will be added to determine
 * appropriate output locations.
 *
 */

#include <ast.h>
#include <quark.h>

// not in quark.h despite manual...
#define NODEP INPUT
#define MEM_SZ (sizeof(void *))

// Add a subexpression to the running queue.
static void add_q(Quark *q, Ast *a, MemSpace *mem, SMap *min, Map *mout, Ast **outp);

int run_quark(Ast *a, MemSpace *mem, void **ret, SMap *named) {
    Map *mout = map_ctor(256, sizeof(void *));
    Quark *q = QUARK_New(0); // nthreads = ncores

    add_q(q, a, mem, named, mout, &a);
    QUARK_Barrier(q);

    *ret = a->val; // extract location of final output tensor
    map_dtor(&mout);
    QUARK_Delete(q);
    return 0;
}

void qalloc_tdot(Quark *q) {
    Ast *op, *a, *b, *c;
    MemSpace *mem;
    // macro
    quark_unpack_args_5(q, mem, op, a, b, c);
    op->val = uniq_tens(mem, c->val);
}
void qrun_tdot(Quark *q) {
    Ast *op;
    MemSpace *mem;

    quark_unpack_args_2(q, mem, op);
    tdot(op->val, op->dot->a->val, op->dot->pa, \
                  op->dot->b->val, op->dot->pb);
    tensor_dtor(&op->dot->a->val, mem);
    tensor_dtor(&op->dot->b->val, mem);
}

void qalloc_tadd(Quark *q) {
    Ast *op, *a, *b;
    MemSpace *mem;
    // macro
    quark_unpack_args_4(q, mem, op, a, b);
    op->val = uniq_tens(mem, a->val);
}
void qrun_tadd(Quark *q) {
    Ast *op;
    MemSpace *mem;

    quark_unpack_args_2(q, mem, op);
    tadd(op->val, op->add->b->val, op->add->perm);
    tensor_dtor(&op->add->b->val, mem);
}

void qalloc_scale(Quark *q) {
    Ast *op, *a;
    MemSpace *mem;
    quark_unpack_args_3(q, mem, op, a);
    op->val = uniq_tens(mem, a->val);
}
void qrun_scale(Quark *q) {
    Ast *op;
    MemSpace *mem;

    quark_unpack_args_2(q, mem, op);
    tscale(op->val, op->scale->s);
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
static void add_q(Quark *q, Ast *a, MemSpace *mem, SMap *min, Map *mout, Ast **outp) {
    Quark_Task_Flags tflags = Quark_Task_Flags_Initializer;
    Ast *b = map_get(mout, &a);

    if(b == (Ast *)mout) {
        printf("Error! input graph contains a cycle: ");
        write_ast_label(stdout, a);
        printf("\n");
        *outp = NULL;
        return;
    } else if(b != NULL) { // ref. already added.
        *outp = b;
        return;
    }

    if(a->type == TRef) {
        map_put(mout, &a, mout); // insert bad ref to detect cycles
        if( (b = smap_get(min, a->ref)) == NULL) {
            printf("Error! Undefined variable, %s\n", a->ref);
            return;
        }
        add_q(q, b, mem, min, mout, outp);
        map_put(mout, &a, *outp);
        return;
    }

    switch(a->type) {
    case TDot:
        add_q(q, a->dot->a, mem, min, mout, &a->dot->a);
        add_q(q, a->dot->b, mem, min, mout, &a->dot->b);
        QUARK_Insert_Task(q, qalloc_tdot, NULL,
                          MEM_SZ, mem, NODEP,
                          a->len, a, INOUT,
                          a->dot->a->len, a->dot->a, INPUT,
                          a->dot->b->len, a->dot->b, INPUT);
        QUARK_Task_Flag_Set(&tflags, TASK_LABEL, (intptr_t)"TDot");
        QUARK_Insert_Task(q, qrun_tdot, &tflags,
                          MEM_SZ, mem, NODEP,
                          a->len, a, INOUT,
                          0); // actually all of the above, but alloc gets it
        break;
    case TAdd:
        add_q(q, a->add->a, mem, min, mout, &a->add->a);
        add_q(q, a->add->b, mem, min, mout, &a->add->b);
        QUARK_Insert_Task(q, qalloc_tadd, NULL,
                          MEM_SZ, mem, NODEP,
                          a->len, a, INOUT,
                          a->add->a->len, a->add->a, INPUT,
                          a->add->b->len, a->add->b, INPUT);
        QUARK_Task_Flag_Set(&tflags, TASK_LABEL, (intptr_t)"TAdd");
        QUARK_Insert_Task(q, qrun_tadd, &tflags,
                          MEM_SZ, mem, NODEP,
                          a->len, a, INOUT,
                          0);
        break;
    case TScale:
        add_q(q, a->scale->a, mem, min, mout, &a->scale->a);
        QUARK_Insert_Task(q, qalloc_scale, NULL,
                          MEM_SZ, mem, NODEP,
                          a->len, a, INOUT,
                          a->scale->a->len, a->scale->a, INPUT);
        QUARK_Task_Flag_Set(&tflags, TASK_LABEL, (intptr_t)"TScale");
        QUARK_Insert_Task(q, qrun_scale, &tflags,
                          MEM_SZ, mem, NODEP,
                          a->len, a, INOUT,
                          0);
        break;
    case TBase:
        if(a->base->type == TTens) {
            a->val = a->base->t; // set output location
                                 // tell allocator to ignore it
            insert_unmanaged(mem, a->val->x, sizeof(double)*a->val->len);
        } else {
            printf("Error! Unknown / non-implemented Base type: %d\n",
                        a->base->type);
            return;
        }
        break;
    default:
        printf("Error! Unknown / non-implemented Ast type: %d\n", a->type);
        return;
    }
    return;
}

