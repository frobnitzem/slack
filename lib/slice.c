#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <lib/slice.h>

// Construct a slice
Slice slice_ctor(size_t width, size_t n, size_t max) {
    Slice s = malloc(sizeof(struct slice_s));
    if(max < n) {
        fprintf(stderr, "fatal: bad slice constructor\n");
        exit(1);
    }
    s->x = malloc(width*max);
    s->width = width;
    s->n = n;
    s->max = max;
    return s;
}

// I don't want to talk about this one.
void slice_dtor(Slice *s) {
    free((*s)->x);
    free(*s);
    *s = NULL;
}

// copy up to min(n)
void slice_copy(Slice dst, Slice src) {
    uint32_t n = dst->n < src->n ? dst->n : src->n;
    if(dst->width != src->width) {
        fprintf(stderr, "fatal: copy between slices of differing width.\n");
        exit(1);
    }
    memcpy(dst->x, src->x, src->width*n);
}

// Append m elements at a time to the end
Slice slice_append(Slice s, void *next, int m) {
    Slice t;
    if(s->max >= s->n+m) { // 2 views of same data
        t = malloc(sizeof(struct slice_s));
        t->width = s->width;
        t->n = s->n+m;
        t->max = s->max;
        t->x = s->x;
    } else {
        t = slice_ctor(s->width, s->n+m, 2*s->max + m);
        memcpy(t->x, s->x, s->width*s->n);
    }
    memcpy(t->x + s->width*s->n, next, s->width*m);
    return t;
}

Slice slice(Slice s, int st, int en) {
    Slice t;
    if(st < 0 || en > s->n || st > en) {
        fprintf(stderr, "fatal: bad slice operation.\n");
        exit(1);
    }
    t = malloc(sizeof(struct slice_s));
    t->x = s->x + s->width*st;
    t->n = en - st;
    t->max = s->max - st;
    t->width = s->width;
    return t;
}

