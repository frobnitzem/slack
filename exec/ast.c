#include <stdlib.h>
#include <string.h>
#include <ast.h>

Ast *mkCons(Ast *car, Ast *cdr) {
    Ast *r = malloc(PAIR_SIZE);
    r->type = TCons;
    r->len = PAIR_SIZE;

    r->pair->a = car; r->pair->b = cdr;
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

    r->type = TDot;
    r->len = DOT_SIZE(ind->n);
    
    r->dot->a = a; r->dot->b = b;
    r->dot->n = ind->n;
    memcpy(r->dot->ind, ind->x, ind->n);
    return r;
}

// Or you can just link a ref by updating a in-place
// (assuming its not referenced anywhere else)
Ast *mkNamed(Ast *ref, Ast *term) {
    uint32_t len = strlen(ref->ref->name)+1;
    Ast *r = malloc(REF_SIZE(ref->ref->n, len));

    r->type = TNamed;
    r->len = REF_SIZE(ref->ref->n, len);

    r->ref->name = (char *)r->ref->ind + ref->ref->n;
    r->ref->n = ref->ref->n;
    memcpy(r->ref->ind, ref->ref->ind, ref->ref->n+len);
    r->ref->a = term;
    return r;
}

Ast *mkRef(char *name, Slice ind) {
    uint32_t len = strlen(name)+1;
    Ast *r = malloc(REF_SIZE(ind->n, len));
    if(ind->width != 1) {
        fprintf(stderr, "fatal: bad index size - should be 1.\n");
        exit(1);
    }

    r->type = TRef;
    r->len = REF_SIZE(ind->n, len);

    r->ref->name = (char *)r->ref->ind + ind->n;
    r->ref->n = ind->n;
    memcpy(r->ref->ind, ind->x, ind->n);
    memcpy(r->ref->name, name, len);
    r->ref->a = NULL;
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
    case TNamed:
        *t = &a->ref->a;
        return 1;
    case TSum:
    case TDiff:
    case TCons:
        *t = &a->pair->a;
        return 2;
    case TDot:
        *t = &a->dot->a;
        return 2;
    case TRef:
        //*t = NULL;
        //return 0;
    default:
        break;
    }
    *t = NULL;
    return 0;
}

