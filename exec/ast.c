#include <stdlib.h>
#include <string.h>
#include <ast.h>

void print_vec(int *x, int n);

Ast *new_ast(size_t len) {
    Ast *r = malloc(len);
    r->len = len;
    return r;
}

Ast *simpScale(double alpha, Ast *a) {
    if(alpha == 1.0)
        return a;

    switch(a->type) {
    case TRef:
        break;
    case TBase:
        if(a->base->type == BZeroTens) {
            return a;
        }
        break;
    case TScale: // Don't need to recurse if tree was made of simpScale.
        a->scale->alpha *= alpha;
        return a;
    case TAdd:
        a->add->alpha *= alpha;
        a->add->beta  *= alpha;
        return a;
    case TDot:
        a->dot->alpha *= alpha;
        a->dot->beta  *= alpha;
        return a;
    }
    // give up.
    return mkScale(alpha, a);
}

Ast *mkScale(double alpha, Ast *a) {
    Ast *r = new_ast(SCALE_SIZE);
    r->type = TScale;

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
        return simpScale(alpha, a);
    if(isZero(a)) {
        if(isId(n, pb)) {
            return simpScale(beta, b);
        }
        // Push transpose down.
        switch(b->type) {
        case TRef:
            break;
        case TBase:
            if(b->base->type == BZeroTens) {
                return b; // permutation is unimportant
            }
            break;
        case TAdd:
            // generic solution.  Note that if b->add->a
            // still has a transpose, it is inevitable (unless the
            // new combination is id)
            if(isZero(b->add->a)) {
            // beta (a 0 + b B^T)^T -> beta b B^T^T
                compose_permut(b->add->n, b->add->pb, n, pb);
                return mkTranspose(beta * b->add->beta, b->add->b,
                                   b->add->n, b->add->pb);
            }
            // beta (a A + b B^T)^T -> beta a A^T + beta b B^T^T
            b->add->alpha *= beta;
            b->add->beta  *= beta;
            b->add->a = mkTranspose(1.0, b->add->a, n, pb);
            compose_permut(b->add->n, b->add->pb, n, pb);
            return b;
        case TDot: // beta (a A.B + b C)^T -> a beta (A.B)^T + b beta C^T
            b->dot->alpha *= beta;
            b->dot->beta  *= beta;
            compose_permut(b->dot->na, b->dot->pa, n, pb);
            compose_permut(b->dot->nb, b->dot->pb, n, pb);
            b->dot->c = mkTranspose(1.0, b->dot->c, n, pb);
            return b;
        case TScale: // Scale would have been simplified already if
                     // we could push the transpose.
            break;
        }
        // give up.
        return mkAdd(0.0, mkZero(), beta, b, n, pb);
    }

    // Neither A nor B are zero
    // alpha A + beta (a AA . AB + 0)^T
    //  -> a beta (AA . AB)^T + alpha A
    if(b->type == TDot && isZero(b->dot->c)) { // FMA case
        // reorder output indices of b->dot to match a
        compose_permut(b->dot->na, b->dot->pa, n, pb);
        compose_permut(b->dot->nb, b->dot->pb, n, pb);
        b->dot->alpha *= beta;
        b->dot->beta = alpha;
        b->dot->c = a;
        return b;
    }
    // alpha (a AA . AB + 0) + beta B^T
    //  -> a alpha (AA . AB) + beta B^T
    if(a->type == TDot && isZero(a->dot->c)) {
        a->dot->alpha *= alpha;
        a->dot->beta = beta;
        a->dot->c = mkTranspose(1.0, b, n, pb);
        return a;
    }

    // default
    return mkAdd(alpha, a, beta, b, n, pb);
}

Ast *mkAdd(const double alpha, Ast *a,
           const double beta,  Ast *b, const int n, const uint8_t *pb) {
    Ast *r = new_ast(ADD_SIZE(n));
    r->type = TAdd;

    r->add->a = a;         r->add->b = b;
    r->add->alpha = alpha; r->add->beta = beta;
    r->add->n = n;
    memcpy(r->add->pb, pb, n);
    return r;
}

Ast *mkTranspose(const double alpha, Ast *b, const int n, const uint8_t *perm) {
    //return mkAdd(0.0, mkZero(), alpha, b, n, perm);
    return simpAdd(0.0, mkZero(), alpha, b, n, perm);
}

Ast *mkDot(const double alpha, Ast *a, const int na, const uint8_t *pa,
                               Ast *b, const int nb, const uint8_t *pb,
           const double beta,  Ast *c, const int nc) {
    Ast *r = new_ast(DOT_SIZE(na,nb));

    r->type = TDot;
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
    Ast *r = new_ast(REF_SIZE(len));

    r->type = TRef;
    memcpy(r->ref, name, len);
    return r;
}

Ast *mkZero() {
    Ast *r = new_ast(BASE_SIZE);
    r->type = TBase;
    r->base->type = BZeroTens;
    return r;
}

Ast *mkLit(const int n, const int *shape, double *x) {
    Ast *r = new_ast(T_SIZE(n));
    int i;

    r->type  = TBase;
    r->base->type = BTens;
    r->base->t->n = n;
    r->base->t->x = x;
    r->base->t->len = 1;
    for(i=0; i<n; i++) {
        r->base->t->len *= shape[i];
    }
    memcpy(r->base->t->shape, shape, sizeof(int)*n);
    return r;
}

Ast *mkRand(Slice shape) {
    Ast *r = mkLit(shape->n, (int *)shape->x, NULL);
    int len = r->base->t->len;
    double *x = malloc(sizeof(double)*len);
    int two = 2;
    int seed[4] = {4, 13, 28, 47};

    r->base->t->x = x; // fill with random numbers.
    dlarnv_(&two, seed, &len, x);

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
