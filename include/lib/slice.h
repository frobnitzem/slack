#ifndef _SLICE_H
#define _SLICE_H

#include <stdlib.h>

struct slice_s {
    void *x;
    uint32_t n;
    uint32_t max;
    uint32_t width;
};

typedef struct slice_s *Slice;

Slice slice_ctor(size_t width, size_t n, size_t max);
void slice_dtor(Slice *s);
void slice_copy(Slice dst, Slice src);
Slice slice_append(Slice s, void *next, int m);
Slice slice(Slice s, int st, int en);

#endif
