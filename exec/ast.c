#include <stdlib.h>
#include <string.h>
#include <ast.h>

Ast *mkScale(Ast *a, float scale) {
    Ast *r = malloc(SCALE_SIZE);
    r->type = TScale;
    r->len = SCALE_SIZE;

    r->scale->a = a;
    r->scale->s = scale;
    return r;
}

Ast *mkSum(Ast *a, Ast *b) {
    Ast *r = malloc(PAIR_SIZE);
    r->type = TSum;
    r->len = PAIR_SIZE;

    r->pair->a = a; r->pair->b = b;
    return r;
}

Ast *mkDiff(Ast *a, Ast *b) { // a - b
    Ast *r = malloc(PAIR_SIZE);
    r->type = TDiff;
    r->len = PAIR_SIZE;

    r->pair->a = a; r->pair->b = b;
    return r;
}

Ast *mkDot(Ast *a, Ast *b, Slice ind) {
    Ast *r = malloc(DOT_SIZE(ind->n));
    if(ind->width != 1) {
        fprintf(stderr, "fatal: bad index size - should be 1.\n");
        exit(1);
    }
    if(ind->n % 2 != 0) { // ind contains pairs of uint8_t
        fprintf(stderr, "fatal: mkDot called with odd number of indices?\n");
        exit(1);
    }

    r->type = TDot;
    r->len = DOT_SIZE(ind->n);
    
    r->dot->a = a; r->dot->b = b;
    r->dot->n = ind->n/2;
    memcpy(r->dot->ind, ind->x, ind->n);
    return r;
}

Ast *mkRef(char *name) {
    uint32_t len = strlen(name)+1;
    Ast *r = malloc(REF_SIZE(len));

    r->type = TRef;
    r->len = REF_SIZE(len);

    memcpy(r->ref, name, len);
    return r;
}

Ast *mkTranspose(Ast *a, int n, uint8_t *perm) {
    uint32_t len = T_SIZE(n);
    Ast *r = malloc(len);
    r->type = TTranspose;
    r->len = len;
    r->t->a = a;
    r->t->n = n;
    memcpy(r->t->perm, perm, n);
    return r;
}

Ast *mkLit(Tensor *t) {
    Ast *r = malloc(BASE_SIZE);

    r->type  = TBase;
    r->len   = BASE_SIZE;
    r->base->type = TTens;
    r->base->t = t;
    return r;
}

char *estrdup(char *a) {
    char *ret;
    if(a == NULL) return NULL;
    if( (ret = strdup(a)) == NULL) {
        fprintf(stderr, "OOM - terminating!");
        exit(1);
    }
    return ret;
}

int ast_children(Ast *a, Ast ***t) {
    switch(a->type) {
    case TTranspose:
        *t = &a->t->a;
        return 1;
    case TScale:
        *t = &a->scale->a;
        return 1;
    case TReduce:
        *t = &a->reduce->a;
        return 1;
    case TSum:
    case TDiff:
        *t = &a->pair->a;
        return 2;
    case TDot:
        *t = &a->dot->a;
        return 2;
    case TBase:
    case TRef:
        //*t = NULL;
        //return 0;
    default:
        break;
    }
    *t = NULL;
    return 0;
}

