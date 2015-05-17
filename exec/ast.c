#include <stdlib.h>
#include <string.h>
#include <ast.h>

void print_vec(int *x, int n);

Ast *mkScale(double alpha, Ast *a) {
    Ast *r = malloc(SCALE_SIZE);
    r->type = TScale;
    r->len = SCALE_SIZE;

    r->scale->a = a;
    r->scale->alpha = alpha;
    return r;
}

static int isId(const int n, const uint8_t *perm) {
    int i;
    for(i=0; i<n; i++)
        if(perm[i] != i)
            return 0;
    return 1;
}

#define isZero(a) (a->type == TBase && a->base->type == BZeroTens)

// Check for simplifications.
Ast *simpAdd(const double alpha, Ast *a,
             const double beta,  Ast *b, const int n, const uint8_t *pb) {
    uint8_t *perm;

    if(isZero(b))
        return a;
    if(isZero(a) && isId(n, pb)) {
        return b;
    }
    if(b->type == TDot && isZero(b->dot->c)) { // FMA case
        // reorder output indices of b->dot to match a
        perm = inv_permut(n, pb);
        compose_permut(b->dot->na, b->dot->pa, n, perm);
        compose_permut(b->dot->nb, b->dot->pb, n, perm);
        free(perm);
        b->dot->alpha *= beta;
        b->dot->beta = alpha;
        b->dot->c = a;
        return b;
    }
    if(a->type == TDot && isZero(a->dot->c)) { // FMA case
        // reorder output indices of a->dot to match b
        compose_permut(a->dot->na, a->dot->pa, n, pb);
        compose_permut(a->dot->nb, a->dot->pb, n, pb);
        a->dot->alpha *= alpha;
        a->dot->beta = beta;
        a->dot->c = b;
        // transpose final result
        return mkTranspose(alpha, a, n, pb);
    }

    // default
    return mkAdd(alpha, a, beta, b, n, pb);
}

Ast *mkAdd(const double alpha, Ast *a,
           const double beta,  Ast *b, const int n, const uint8_t *pb) {
    uint32_t len = ADD_SIZE(n);
    Ast *r = malloc(len);
    r->type = TAdd;
    r->len = len;

    r->add->a = a;         r->add->b = b;
    r->add->alpha = alpha; r->add->beta = beta;
    r->add->n = n;
    memcpy(r->add->pb, pb, n);
    return r;
}

Ast *mkTranspose(const double alpha, Ast *b, const int n, const uint8_t *perm) {
    return mkAdd(0.0, mkZero(), alpha, b, n, perm);
}

Ast *mkDot(const double alpha, Ast *a, const int na, const uint8_t *pa,
                               Ast *b, const int nb, const uint8_t *pb,
           const double beta,  Ast *c, const int nc) {
    uint32_t len = DOT_SIZE(na,nb);
    Ast *r = malloc(len);

    r->type = TDot;
    r->len = len;
    r->dot->pb = r->dot->pa + na;

    r->dot->a = a; r->dot->b = b; r->dot->c = c;
    r->dot->alpha = alpha; r->dot->beta = beta;
    r->dot->na = na; memcpy(r->dot->pa, pa, na);
    r->dot->nb = nb; memcpy(r->dot->pb, pb, nb);
    r->dot->nc = nc;
    return r;
}

// Simplified form where output indices are ordered:
// (leftover A) (x) (leftover B)
// and zero is added.
Ast *mkTensDot(const double alpha, Ast *a, const int na,
                                   Ast *b, const int nb, Slice ind) {
    uint8_t *pa, *pb;
    uint8_t *x = ind->x;
    Ast *r;
    int i, j;

    if(ind->width != 1) {
        fprintf(stderr, "fatal: bad index size - should be 1 byte.\n");
        exit(1);
    }
    if(ind->n % 2 != 0) { // ind contains pairs of uint8_t
        fprintf(stderr, "fatal: mkDot called with odd number of indices?\n");
        exit(1);
    }
    pa = malloc(na+nb);
    pb = pa + na;

    for(i=0; i<na; i++) { // will number sequentially later
        pa[i] = 255;
    }
    for(i=0; i<nb; i++) {
        pb[i] = 255;
    }

    // number contracted indices first
    j = na + nb - ind->n; // first contraction index
    for(i=0; i<ind->n; i+=2) {
        pa[x[i+0]] = j;
        pb[x[i+1]] = j++;
    }

    j = 0;
    for(i=0; i<na; i++) { // A indices go first
        if(pa[i] == 255) {
            pa[i] = j++;
        }
    }
    for(i=0; i<nb; i++) { // B indices go second
        if(pb[i] == 255) {
            pb[i] = j++;
        }
    }
    if(j != na + nb - ind->n) {
        fprintf(stderr, "Error! Bad indices sent to mkTensDot!\n");
    }

    r = mkDot(alpha, a, na, pa, b, nb, pb, 0.0, mkZero(), na + nb - ind->n);
    free(pa);

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

Ast *mkZero() {
    Ast *r = malloc(BASE_SIZE);
    r->type = TBase;
    r->len = BASE_SIZE;
    r->base->type = BZeroTens;
    return r;
}

Ast *mkLit(const int n, const int *shape, double *x) {
    Ast *r = malloc(T_SIZE(n));

    r->type  = TBase;
    r->len   = BASE_SIZE;
    r->base->type = BTens;
    r->base->t->n = n;
    r->base->t->x = x;
    memcpy(r->base->t->shape, shape, sizeof(int)*n);
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
    case TScale:
        *t = &a->scale->a;
        return 1;
    case TAdd:
        *t = &a->add->a;
        return 2;
    case TDot:
        *t = &a->dot->a;
        return 3;
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
