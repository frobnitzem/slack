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

void display_ast(FILE *f, Ast *a, int pri) {
    switch(a->type) {
    case TAdd:
        if(pri > 100) {
            fprintf(f, "(");
        }
        fprintf(f, "tadd ");
        display_ind(f, a->add->n, a->add->pb);

        fprintf(f, " %f ", a->add->alpha);
        display_ast(f, a->add->a, 101);

        fprintf(f, " %f ", a->add->beta);
        display_ast(f, a->add->b, 101);

        if(pri > 100) {
            fprintf(f, ")");
        }
        break;
    case TRef:
        fprintf(f, "%s", a->ref);
        break;
    case TDot:
        if(pri > 100) {
            fprintf(f, "(");
        }
        fprintf(f, "tdot %f ", a->dot->beta);
        display_ast(f, a->dot->c, 101);
        fprintf(f, " %f ", a->dot->alpha);
        display_ast(f, a->dot->a, 101);
        fprintf(f, " ");
        display_ind(f, a->dot->na, a->dot->pa);
        fprintf(f, " ");
        display_ast(f, a->dot->b, 101);
        fprintf(f, " ");
        display_ind(f, a->dot->nb, a->dot->pb);
        if(pri > 100) {
            fprintf(f, ")");
        }
        break;
    case TScale:
        if(pri > 100) {
            fprintf(f, "(");
        }
        fprintf(f, "tscale %f ", a->scale->alpha);
        display_ast(f, a->scale->a, 101);
        if(pri > 100) {
            fprintf(f, ")");
        }
        break;
    case TBase:
        if(a->base->type == BZeroTens) {
            fprintf(f, "Zero");
            break;
        }
    default:
        fprintf(f, "(invalid Ast type %d)", a->type);
    }
}

