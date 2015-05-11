#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include <ast.h>

// Internally, this function is used.
void display_ast(FILE *f, Ast *a, int pri);

void write_ast(FILE *f, Ast *a) {
    display_ast(f, a, 0);
    fprintf(f, "\n");
}

void display_ind(FILE *f, int n, uint8_t *x) {
    int i;

    fprintf(f, "(");
    for(i=0; i<n; i++) {
        fprintf(f, "%u,", x[i]);
    }
    fprintf(f, ")");
}

void display_ind2(FILE *f, int n, uint8_t *x) {
    int i;
    if(n == 0) {
        fprintf(f, "[]");
        return;
    }

    fprintf(f, "[");
    for(i=0; i<n-1; i++,x+=2) {
        fprintf(f, "(%u,%u), ", x[0], x[1]);
    }
    fprintf(f, "(%u,%u)]", x[0], x[1]);
}

// Binary primitive operation.
void display_prim(void *f, char *bop, struct Pair *p, int pri) {
    int bp = 2; // precedence of binary operation

    if(pri > bp) {
        fprintf(f, "(");
    }
    display_ast(f, p->a, bp);
    fprintf(f, " %s ", bop);
    display_ast(f, p->b, bp);
    if(pri > bp) {
        fprintf(f, ")");
    }
    return;
}

void display_ast(FILE *f, Ast *a, int pri) {
    switch(a->type) {
    case TSum:
        display_prim(f, "`+t`", a->pair, pri);
        break;
    case TDiff:
        display_prim(f, "`-t`", a->pair, pri);
        break;
    case TRef:
        fprintf(f, "%s", a->ref);
        break;
    case TDot:
        if(pri > 100) {
            fprintf(f, "(");
        }
        fprintf(f, "tensdot ");
        display_ast(f, a->dot->a, 101);
        fprintf(f, " ");
        display_ast(f, a->dot->b, 101);
        fprintf(f, " ");
        display_ind2(f, a->dot->n, a->dot->ind);
        if(pri > 100) {
            fprintf(f, ")");
        }
        break;
    case TTranspose:
        if(pri > 100) {
            fprintf(f, "(");
        }
        fprintf(f, "transpose ");
        fprintf(f, " ");
        display_ast(f, a->t->a, 101);
        fprintf(f, " ");
        display_ind(f, a->t->n, a->t->perm);
        if(pri > 100) {
            fprintf(f, ")");
        }
        break;
    case TScale:
        if(pri > 100) {
            fprintf(f, "(");
        }
        fprintf(f, "tscale %f ", a->scale->s);
        display_ast(f, a->scale->a, 101);
        if(pri > 100) {
            fprintf(f, ")");
        }
        break;
    default:
        fprintf(f, "(invalid Ast type %d)", a->type);
    }
}

