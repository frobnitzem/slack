/*  Copyright David M. Rogers, 2015
 *  This code is made available under terms of the GNU GPL.
 *  A copy is included at the top-level of the source tree.
 */
#include <sys/types.h>
#include <sys/wait.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <ast.h>
#include <parser.h>


#define usage() { \
    fprintf(stderr, "usage: %1$s line.tex\n", argv[0]); \
    exit(1); \
}

static void show_assign(const char *key, void *value, void *ignored) {
    Ast *a = (Ast *)value;
    char *name;

    printf("Showing '%s':\n", key);
    write_ast(stdout, a);
    asprintf(&name, "%s.dot", key);
    show_ast(name, a, 1);
    free(name);
}

int main(int argc, char *argv[]) {
    struct Environ e = {
        .debuglevel = 1,
    };
    FILE *f;
    SMap *defs;
    int n;
    Ast *a;

    if(argc != 2) {
        usage();
    }

    if( (f = fopen(argv[1], "r")) == NULL) {
            perror("Error opening file");
            return -1;
    }

    if( (defs = tce2_parse_inp(&e, f)) == NULL) {
        printf("bad parse.\n");
    } else {
        n = smap_iter(defs, show_assign, NULL);
        printf("Total assignments = %d\n", n);
    }

    // execute dag
    if( (a = smap_get(defs, "R")) != NULL) {
        Tensor *t;
        MemSpace *mem = memspace_ctor(32, 1<<30); // 1 Gb
        if(mem == NULL) {
            printf("Error constructing memspace.\n");
            return 1;
        }
        if( (t = run_quark(a, mem, defs)) == NULL) {
            printf("Error executing dag.\n");
            return 1;
        }
        printf("Result = \n");
        print_tens(stdout, t);
        printf("Used mem = %lu\n", mem->used);
        memspace_dtor(&mem);
    }

    smap_dtor(&defs);
 
    return 0;
}

