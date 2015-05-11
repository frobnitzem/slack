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

void write_ast_label(FILE *f, Ast *a) {
    switch(a->type) {
    case TScale:
        fprintf(f, "x %.2f", a->scale->s);
        break;
    case TTranspose:
        fprintf(f, "transpose ");
        write_list(f, a->t->n, a->t->perm);
        break;
    case TRef:
        fprintf(f, "%s", a->ref);
        break;
    case TSum:
        fprintf(f, "+");
        break;
    case TDiff:
        fprintf(f, "-");
        break;
    case TDot:
        fprintf(f, "*");
        write_list2(f, a->dot->n, a->dot->ind);
        break;
    case TBase:
        if(a->base->type == TTens) {
            fprintf(f, "Tensor ");
            write_int_list(f, a->base->t->n, a->base->t->shape);
        } else {
            fprintf(f, "unknown Base");
        };
        break;
    default:
        fprintf(f, "unk");
    }
}

static void show_dot_node(FILE *f, Ast *a, int n) {
    /*if(a == &ast_nil) {
        fprintf(f, "\"%p\" [label=\"nil\"]\n");
    }*/
    fprintf(f, "\"%p\" [label=\"", a);
    write_ast_label(f, a);
    fprintf(f, "\" rank=%d];\n", n);
}

static void dot_rec(FILE *f, Map *m, Ast *a, int n) {
    Ast **t;
    int i, nch;

    if(map_get(m, &a) != NULL)
        return;

    map_put(m, &a, a);
    show_dot_node(f, a, n);

    nch = ast_children(a, &t);
    for(i=0; i<nch; i++) {
        if(t[i] == NULL) continue;
        fprintf(f, "\"%p\" -> \"%p\";\n", a, t[i]);
    }
    for(i=0; i<nch; i++) {
        if(t[i] == NULL) continue;
        dot_rec(f, m, t[i], n+1);
    }
}

void ast_to_dot(FILE *f, Ast *a) {
    Map *m;

    if( (m = map_ctor(16, sizeof(Ast *))) == NULL) {
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

