/*    Copyright (C) David M. Rogers, 2015
 *    
 *    David M. Rogers <predictivestatmech@gmail.com>
 *    Nonequilibrium Stat. Mech. Research Group
 *    Department of Chemistry
 *    University of South Florida
 *
 *    This file is part of USF-slack.
 *
 *    This version of slack is free software: you can redistribute
 *    it and/or modify it under the terms of the GNU General Public
 *    License as published by the Free Software Foundation, either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    USF-slack is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    in the LICENSE file at the top of USF-slack's source tree.
 *    If not, see <http://www.gnu.org/licenses/>.
 *
 * The memory management strategy of the interpreter deserves some comment.
 * Hand-written code usually uses a minimum number of memory
 * areas, and sends intermediate results to them, as in a circular
 * buffer.  We can accomplish the same thing here by maintaining
 * a list of active memory spaces during interpretation.
 *
 * The resulting exec function has type:
 *
 * Ast_Tensor -> ST MemSpace Ast_Lit_Tensor
 *
 * The binding expressions are refcounted.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <memspace.h>

/****************** Block management routines. ******************/
#define elock(mtx) \
    { int result_var; \
      if( (result_var = pthread_mutex_lock(&(mtx))) != 0) { \
         fprintf(stderr, "MemSpace lock error: %s\n", strerror(result_var)); \
         exit(1); \
      } }
#define eunlock(mtx) \
    { int result_var; \
      if( (result_var = pthread_mutex_unlock(&(mtx))) != 0) { \
         fprintf(stderr, "MemSpace unlock error: %s\n", strerror(result_var)); \
         exit(1); \
      } }

struct Block *block_ctor(int nref, size_t sz) {
    int r;
    struct Block *b = malloc(sizeof(struct Block));

    if(pthread_mutex_init(&b->lock, NULL)) {
        fprintf(stderr, "Block mutex error: %s\n", strerror(r));
        exit(1);
    }
    b->nref = nref;
    b->sz = b->used = sz;
    if(nref == 0)
        b->used = 0;

    return b;
}

static void block_dtor(const void *key, void *val, void *info) {
    void *x = *(void **)key;
    struct Block *b = val;
    if(b->nref > 0) {
        printf("Manged block %p left with nref=%d, sz=%lu, used=%lu\n",
                    x, b->nref, b->sz, b->used);
    }
    if(b->nref >= 0) { // managed
        int r;
        free(x);
        if( (r = pthread_mutex_destroy(&b->lock)) != 0) {
            fprintf(stderr, "Block dtor error: %s\n", strerror(r)); \
                      exit(1);
        }
    }

    free(b);
}

MemSpace *memspace_ctor(int n, size_t max) {
    int r;
    MemSpace *mem = malloc(sizeof(MemSpace));
    if(n < 16)
        n = 16;

    mem->m = map_ctor(n, sizeof(void *));
    mem->max = max > 1024 ? max : 1024; // not currently enforced
    mem->used = 0;
    mem->recent = NULL; mem->recent_x = NULL;

    if( (r = pthread_mutex_init(&mem->lock, NULL)) != 0) {
        fprintf(stderr, "MemSpace mutex error: %s\n", strerror(r));
        exit(1);
    }
    if( (r = pthread_cond_init(&mem->avail, NULL)) != 0) {
        fprintf(stderr, "MemSpace cond error: %s\n", strerror(r));
        exit(1);
    }
    return mem;
}

// display un-free()-d memory
void memspace_dtor(MemSpace **memp) {
    int r;
    elock((*memp)->lock);
    map_iter((*memp)->m, block_dtor, NULL);
    map_dtor(&(*memp)->m);
    eunlock((*memp)->lock);

    if( (r = pthread_mutex_destroy( &(*memp)->lock )) != 0) {
        fprintf(stderr, "MemSpace dtor error: %s\n", strerror(r));
        exit(1);
    }
    if( (r = pthread_cond_destroy( &(*memp)->avail )) != 0) {
        fprintf(stderr, "MemSpace dtor error: %s\n", strerror(r));
        exit(1);
    }
    *memp = NULL;
}

struct best_res {
    size_t sz;
    size_t minsz;
    void *best;
    struct Block *bb;
};

// Leaves the "best" solution locked once found.
static void find_best(const void *key, void *val, void *info) {
    struct best_res *r = info;
    struct Block *b = val;

    elock(b->lock);
    if(!b->nref && b->sz >= r->sz &&
            (r->best == NULL || b->sz < r->minsz)) {
        if(r->bb != NULL)
            eunlock(r->bb->lock);
        r->minsz = b->sz;
        r->best = *(void **)key;
        r->bb = b;
    } else {
        eunlock(b->lock);
    }
}

void *reserve_block(MemSpace *mem, size_t sz, int nref) {
    struct best_res r = {
        .sz = sz,
        .minsz = 0,
        .best = NULL,
        .bb = NULL,
    };

    if(nref < 1) {
        fprintf(stderr, "reserve_block called with nref = %d!\n", nref);
        exit(1);
    }

    elock(mem->lock);
    map_iter(mem->m, find_best, &r);
    // Deal with OOM condition:
    while(r.best == NULL && mem->used + sz > mem->max) {
        printf("Waiting for %lu bytes (cur = %lu, max = %lu)\n",
                sz, mem->used, mem->max);
        pthread_cond_wait(&mem->avail, &mem->lock);
        // test mem->recent
        elock(mem->recent->lock);
        if(!mem->recent->nref && mem->recent->sz < r.minsz) { // ok.
            r.best = mem->recent_x;
            r.bb = mem->recent;
            mem->recent = NULL; mem->recent_x = NULL;
        } else {
            eunlock(mem->recent->lock);
        }
    }

    if(r.best != NULL) {
        r.bb->nref = nref;
        eunlock(r.bb->lock);
    } else {
        mem->used += sz;
        r.best = malloc(sz);
        struct Block *b = block_ctor(nref, sz);

        map_put(mem->m, &r.best, b);
    }
    eunlock(mem->lock);

    return r.best;
}

void insert_unmanaged(MemSpace *mem, void *buf, size_t sz) {
    struct Block *b = block_ctor(-1, sz);

    elock(mem->lock);
    map_put(mem->m, &buf, b);
    eunlock(mem->lock);
}

/* Get a unique copy of a block (make mutable).
 * sz is a quick optimization in case you don't need to
 * copy the whole block.  It's usually a good idea.
 * 
 * If sz < 0, the used part of the whole block will be copied.
 *
 * The caller can tell if x was duplicated, since a more differenter
 * pointer will be returned.
 *
 */
void *uniq_block(MemSpace *mem, void *x, ssize_t sz, const int nref) {
    struct Block *b;
    void *y;

    elock(mem->lock);
    if( (b = map_get(mem->m, &x)) == NULL) {
        fprintf(stderr, "Error! Memory at %p is unknown to MemSpace.\n", x);
        return NULL;
    }
    eunlock(mem->lock);

    elock(b->lock);
    if(b->nref == 1) { // the whole point.
        eunlock(b->lock);
        return x;
    }

    if(sz < 0)
        sz = b->used;
    eunlock(b->lock);

    // need to copy.
    y = reserve_block(mem, b->used, nref);
    memcpy(y, x, sz);

    elock(b->lock);
    b->nref--; // decrement refcount
    eunlock(b->lock);

    return y;
}

// Release block (note this must be called multiple times if nref > 1)
// It also frees 'info' if the block is left unreferenced.
void release_block_if(MemSpace *mem, void *x, void **info) {
    int n = -1;
    struct Block *b;

    elock(mem->lock);
    if( (b = map_get(mem->m, &x)) == NULL) {
        fprintf(stderr, "Error! Memory at %p is unknown to MemSpace.\n", x);
        eunlock(mem->lock);
        return;
    }
    eunlock(mem->lock);

    elock(b->lock);
    n = b->nref;
    if(n > 0)
        b->nref--;
    eunlock(b->lock);

    if(n == 0) {
        fprintf(stderr, "Double-free of managed block %p of"
                        " sz %lu (%lu used)!\n", x, b->sz, b->used);
    } else if(n == 1) {
        if(info != NULL) {
            free(*info); // Hack the planet!
            *info = NULL;
        }
        // Inform memspace of newly released mem.
        elock(mem->lock);
        mem->recent = b;
        mem->recent_x = x;
        eunlock(mem->lock);
        pthread_cond_broadcast(&mem->avail); // check all waiting spaces
    }
}

void release_block(MemSpace *mem, void *x) {
    release_block_if(mem, x, NULL);
}

