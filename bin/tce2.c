#include <sys/types.h>
#include <sys/wait.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ast.h"
#include "parser.h"


#define usage() { \
    fprintf(stderr, "usage: %1$s line.tex\n", argv[0]); \
    exit(1); \
}

int main(int argc, char *argv[]) {
    struct Environ e = {
        .debuglevel = 1,
    };
    Ast *a;
    FILE *f;

    if(argc != 2) {
        usage();
    }

    if( (f = fopen(argv[1], "r")) == NULL) {
            perror("Error opening file");
            return -1;
    }

    e.ind_map = smap_ctor(32);
    e.nindices = 0;
    a = tce2_parse_inp(&e, f);
    printf("Total: %d indices.\n", e.nindices);
    show_ast("the_ast.dot", a, 1);
    smap_dtor(&e.ind_map);
 
    return 0;
}

