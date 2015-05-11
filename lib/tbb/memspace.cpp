// Copyright David M. Rogers
// Released into Public Domain
#include "memspace.h"

using namespace tbb;

extern "C" {
    #include "tens.h"

    // Construct the tensor header (t->x = NULL);
    Tensor *tensor_ctor(int nd, int *shape) {
        int i;
        // These headers are leaked and I don't care.
        Tensor *t = (Tensor *)malloc(sizeof(Tensor)+sizeof(int)*nd);

        t->len = 1;
        for(i=0; i<nd; i++)
            t->len *= shape[i];

        memcpy(t->shape, shape, sizeof(int)*nd);
        t->x = NULL;

        return t;
    }

    // Don't call remove_unmanaged, since tensor_dtor
    // is called on fake copies of input data during execution.
    void tensor_dtor(Tensor **t, void *m) {
        MemSpace *mem = (MemSpace *)m;
        if(m != NULL) {
            mem->release(*t);
        } else {
            // Can't free info if using memory management...
            // (hence the leak above).
            free(*t);
            *t = NULL;
        }
    }
    
    // Allocate a new tensor.
    // x will be managed by MemSpace.
    Tensor *mkTensor(int nd, int *shape, int nref, void *m) {
        MemSpace *mem = (MemSpace *)m; // Did I stutter?
        Tensor *t = tensor_ctor(nd, shape);
        t->x = mem->reserve(sizeof(double)*t->len, nref, t);
        return t;
    }

    // Cast a block of mem. to a tensor.
    // x won't be managed by MemSpace.
    Tensor *toTensor(double *x, int nd, int *shape, void *m) {
        MemSpace *mem = (MemSpace *)m;
        Tensor *t = tensor_ctor(nd, shape);
        t->x = (void *)x;
        mem->insert_unmanaged((void *)x, sizeof(double)*t->len, t);
        return t;
    }
}

// display un-free()-d memory
MemSpace::~MemSpace() {
    for(MemTbl::iterator it = mem.begin(); it != mem.end(); ++it) {
        if(it->second.nref > 0) {
            printf("Manged block %p left with nref=%d\n", it->first, it->second.nref);
        }
        if(it->second.nref >= 0) // managed
            free(it->first);
    }
}

void *MemSpace::reserve(size_t sz, int nref, void *info) {
    size_t minsz = 0;
    void *best = NULL;
    MemTbl::accessor b;
    int i;

    if(nref < 1) {
        fprintf(stderr, "MemSpace::reserve called with nref = %d!\n", nref);
        exit(1);
    }

    { mutex::scoped_lock lock(lck);
    for(MemTbl::iterator it = mem.begin(); it != mem.end(); ++it) {
        if(!it->second.nref && it->second.sz >= sz &&
                (!minsz || it->second.sz < minsz)) {
            minsz = it->second.sz;
            best = it->first;
        }
    }
    if(best != NULL) {
        mem.find(b, best);
        b->second.nref = nref;
        b.release();
    }
    } // unlock global

    if(best == NULL) {
        void *buf = malloc(sz);
        mem.insert(b, buf);
        b->second.nref = nref;
        b->second.info = info;
        b->second.sz = sz;
        b.release();
    }

    return best;
}

void MemSpace::insert_unmanaged(void *buf, size_t sz, void *info) {
    MemTbl::accessor b;

    mem.insert(b, buf);
    b->second.nref = -1;
    b->second.info = info;
    b->second.sz = sz;
    b.release();
}

/* Get a unique copy of a block (make mutable). */
void *MemSpace::uniq(void *x, size_t sz) {
    MemTbl::accessor b;
    mem.find(b, x);

    if(b->second.nref == 1) { // the whole point.
        b.release();
        return x;
    }

    // need to copy.
    void *y = reserve(sz, 1, b->second.info);
    memcpy(y, x, sz);
    b->second.nref--; // decrement refcount
    b.release();

    return y;
}

// Release block (note this must be called multiple times if nref > 1)
void MemSpace::release(void *x) {
    MemTbl::accessor b;
    mem.find(b, x);
    if(b->second.nref >= 0) {
        if(b->second.nref == 0) {
            fprintf(stderr, "Double-free of managed block %p of sz %lu!\n",
                            x, b->second.sz);
        } else {
            --b->second.nref;
        }
    }
    b.release();
}

// get ref. to header info.
void *MemSpace::get_info(void *x, int *nref, size_t *sz) {
    void *info;
    MemTbl::accessor b;
    mem.find(b, x);
    info = b->second.info;
    *nref = b->second.nref;
    *sz = b->second.sz;
    b.release();

    return x;
}

