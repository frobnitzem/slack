#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>
#include <ast.h>
#include <serial.h>
#include <lib/map.h>

void write_list(FILE *f, int n, uint8_t *ind) {
    int i;
    if(!n) {
        fprintf(f, "()");
        return;
    }

    fprintf(f, "(");
    for(i=0; i<n-1; i++) {
        fprintf(f, "%u,", ind[i]);
    }
    fprintf(f, "%u)", ind[i]);
}

void write_list2(FILE *f, int n, uint8_t *ind) {
    int i;
    if(!n) {
        fprintf(f, "[]");
        return;
    }

    fprintf(f, "[");
    for(i=0; i<n-1; i++) {
        fprintf(f, "(%u,%u),", ind[2*i], ind[2*i+1]);
    }
    fprintf(f, "(%u,%u)]", ind[2*i], ind[2*i+1]);
}

void write_int_list(FILE *f, int n, int *ind) {
    int i;
    if(!n) {
        fprintf(f, "()");
        return;
    }

    fprintf(f, "(");
    for(i=0; i<n-1; i++) {
        fprintf(f, "%d,", ind[i]);
    }
    fprintf(f, "%d)", ind[i]);
}

static void show_dot_node(FILE *f, Ast *a, int rank) {
    Ast **t;
    int i, nch;

    fprintf(f, "\"%p\" [label=\"", a);
    switch(a->type) {
    case TScale:
        fprintf(f, "\" rank=%d];\n"
                   "\"%p\" -> \"%p\" [label=\"%.1f\"];\n", rank,
                       a, a->scale->a, a->scale->alpha);
        return;
    case TAdd:
        fprintf(f, "+\" rank=%d];\n"
                   "\"%p\" -> \"%p\" [taillabel=\"%.1f\"];\n", rank,
                       a, a->add->a, a->add->alpha);
        fprintf(f, "\"%p\" -> \"%p\" [taillabel=\"%.1f\" label=\"",
                       a, a->add->b, a->add->alpha);
        write_list(f, a->add->n, a->add->pb);
        fprintf(f, "\"];\n");
        return;
    case TDot: // TODO: these should really be edge labels...
        if(a->dot->alpha != 1.0) {
          fprintf(f, "[%d]\\n* %.1f\" rank=%d];\n",
                  a->dot->nc, a->dot->alpha, rank);
        } else {
          fprintf(f, "[%d]\\n*\" rank=%d];\n", a->dot->nc, rank);
        }
        fprintf(f,"\"%p\" -> \"%p\" [taillabel=\"%.1f\"];\n",
                     a,    a->dot->c,      a->dot->beta);

        fprintf(f, "\"%p\" -> \"%p\" [label=\"", a, a->dot->a);
        write_list(f, a->dot->na, a->dot->pa);
        fprintf(f, "\"];\n");

        fprintf(f, "\"%p\" -> \"%p\" [label=\"", a, a->dot->b);
        write_list(f, a->dot->nb, a->dot->pb);
        fprintf(f, "\"];\n");
        return;
    case TRef:
        fprintf(f, "%s", a->ref);
        break;
    case TBase:
        if(a->base->type == BTens) {
            fprintf(f, "Tensor ");
            write_int_list(f, a->base->t->n, a->base->t->shape);
        } else if(a->base->type == BZeroTens) {
            fprintf(f, "Zero");
        } else {
            fprintf(f, "unknown Base");
        };
        break;
    default:
        fprintf(f, "unk");
    }
    fprintf(f, "\" rank=%d];\n", rank);

    // default is to not label links
    nch = ast_children(a, &t);
    for(i=0; i<nch; i++) {
        if(t[i] == NULL) continue;
        fprintf(f, "\"%p\" -> \"%p\";\n", a, t[i]);
    }
}

static void dot_rec(FILE *f, Map *m, Ast *a, int n) {
    Ast **t;
    int nch, i;

    if(map_get(m, &a) != NULL)
        return;

    map_put(m, &a, a);
    show_dot_node(f, a, n);

    nch = ast_children(a, &t);
    for(i=0; i<nch; i++) {
        if(t[i] == NULL) continue;
        dot_rec(f, m, t[i], n+1);
    }
}

void ast_to_dot(FILE *f, Ast *a) {
    Map *m;

    if( (m = map_ctor(32, sizeof(Ast *))) == NULL) {
        return;
    }

    fprintf(f, "digraph Ast {\n");
    if(a != NULL)
        dot_rec(f, m, a, 0);
    fprintf(f, "}\n");

    map_dtor(&m);
}

static void clunk(int signum) {
    while(waitpid(-1, NULL, WNOHANG) > 0);
}

static void ignore_children() {
    struct sigaction act;
    memset(&act, 0, sizeof(struct sigaction));
    act.sa_flags = SA_NOCLDSTOP;
    act.sa_handler = clunk;
    sigaction(SIGCHLD, &act, NULL);
}

int show_ast(char *name, Ast *a, int waitfor) {
    FILE *f;
    char *buf;
    int stat;
    pid_t pid;

    if( (f = fopen(name, "w")) == NULL) {
        perror("Error opening dot output file");
        return 1;
    }
    ast_to_dot(f, a);
    fclose(f);

    //if(ignore_children())
    //    return 1;
    if( (pid = fork()) == 0) {
        //execlp("dotty", "dotty", name, NULL);
        //exit(1);
        if(asprintf(&buf, "dot -Tsvg -o %s.svg %s && inkview %s.svg", name, name, name) < 0) {
            exit(1);
        }
        exit(system(buf));
    }
    if(waitfor) {
        waitpid(pid, &stat, WUNTRACED);
    }
    return 0;
}

