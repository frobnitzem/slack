/*   A simple Ast encoding tensor math expressions.
 */

#ifndef _AST_H
#define _AST_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <lib/slice.h>
#include <lib/smap.h>

#include "tens.h"

typedef struct Ast_s Ast;

// Tuninit is just for uncopied objecs during GC passes.
enum AstType { Tuninit=0, TRef=1, TBase=2,
               TScale=4, TAdd=5, TDot=6 } ;

// ZeroTens does not have base->t set.
enum BaseType { BZeroTens=0, BTens=1, };

/* Ast Types */
// Note that transpose can be implemented via Add
struct Add { // alpha*A[i] + beta*B[f(i)]
    Ast *a, *b;
    double alpha, beta;
    int n; // number of indices
    uint8_t pb[0]; // permutation (transpose indices for b)
};
struct Dot { // alpha (a . b) + beta C
    Ast *a, *b, *c;
    double alpha, beta;
    int na, nb, nc;
    uint8_t *pb;   // permutation of b -- points to end of pa
    uint8_t pa[0]; // permutation of a (see tensdot)
};
struct Scale {
    Ast *a;
    double alpha;
};
struct Base {
    enum BaseType type;
    union {
        Tensor t[0];
    };
};

struct Ast_s {
    enum AstType type;
    uint32_t len;

    union {
        struct Base   base[0];
        struct Dot    dot[0];
        struct Add    add[0];
        struct Scale  scale[0];
        char   ref[0];
    };
};
#define AST_HDR_SIZE    (sizeof(Ast))
#define SCALE_SIZE      (sizeof(Ast) + sizeof(struct Scale))
#define ADD_SIZE(n)     (AST_HDR_SIZE + sizeof(struct Add) + n)
#define DOT_SIZE(n,m)   (AST_HDR_SIZE + sizeof(struct Dot) + n + m)
#define REF_SIZE(n)     (AST_HDR_SIZE + n)

#define BASE_SIZE       (AST_HDR_SIZE + sizeof(struct Base))
#define T_SIZE(n)       (BASE_SIZE + sizeof(Tensor) + sizeof(int)*n)

struct Environ {
    int debuglevel;
};

Ast *simpScale(double alpha, Ast *a);
Ast *mkScale(const double, Ast *);
Ast *simpAdd(const double alpha, Ast *a,
             const double beta,  Ast *b, const int n, const uint8_t *pb);
Ast *mkAdd(const double alpha, Ast *a,
           const double beta,  Ast *b, const int n, const uint8_t *pb);
Ast *mkTranspose(const double alpha, Ast *,
                 const int n, const uint8_t *perm); // wrapper for mkAdd

Ast *mkDot(const double alpha, Ast *a, const int na, const uint8_t *pa,
                               Ast *b, const int nb, const uint8_t *pb,
           const double  beta, Ast *c, const int nc);
Ast *mkTensDot(const double alpha, Ast *a, const int na,
                                   Ast *b, const int nb, Slice ind);

Ast *mkRef(char *name);
Ast *mkLit(const int n, const int *shape, double *x);
Ast *mkRand(Slice shape);
Ast *mkZero(); // zero Ast element

int ast_children(Ast *a, Ast ***t);

char *estrdup(char *);

#include "serial.h"
#include "exec.h"

#endif
