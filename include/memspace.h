#ifndef _MEMSPACE_H
#define _MEMSPACE_H

// included by tens.h
#include <lib/map.h>
#include <pthread.h>

// Memory block info. (dynamically allocated value inside MemSpace.m)
struct Block {
    int nref; // < 0 => copy on uniq()-block, not responsible for free()
    size_t sz, used;
    pthread_mutex_t lock;
};

typedef struct {
    Map *m; // Map (void *) (struct Block *)
    pthread_mutex_t lock;
    size_t max; // max-space
    size_t used; // total used space
} MemSpace;

MemSpace *memspace_ctor(int, size_t max);
void memspace_dtor(MemSpace **mem);

// void *x is the location of the allocated memory block (key for MemSpace.m).
void *reserve_block(MemSpace *mem, size_t sz, int nref);
void insert_unmanaged(MemSpace *mem, void *buf, size_t sz);
void *uniq_block(MemSpace *mem, void *x, ssize_t sz, const int nref);

void release_block(MemSpace *mem, void *x);
// This will release an additional pointer (e.g. a header str.)
// if x's refcount drops to 0.
void release_block_if(MemSpace *mem, void *x, void **info);

#endif
