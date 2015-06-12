/*    Copyright (C) David M. Rogers, 2015
 *    
 *    David M. Rogers <predictivestatmech@gmail.com>
 *    Nonequilibrium Stat. Mech. Research Group
 *    Department of Chemistry
 *    University of South Florida
 *
 *    This file is part of USF-slack.
 *
 *    This version of slack is free software: you can redistribute
 *    it and/or modify it under the terms of the GNU General Public
 *    License as published by the Free Software Foundation, either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    USF-slack is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    in the LICENSE file at the top of USF-slack's source tree.
 *    If not, see <http://www.gnu.org/licenses/>.
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
    fprintf(stderr, "usage: %1$s [-v] [-n 1] line.tex\n", name); \
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
        .debuglevel = 0,
    };
    char *name = argv[0];
    FILE *f;
    SMap *defs;
    int n, nthreads=0;
    Ast *a;

    while(1) {
        if(argc < 2) {
            usage();
        }
        if(!strcmp(argv[1], "-v")) {
            e.debuglevel = 1;
            argv++;
        } else if(!strcmp(argv[1], "-n")) {
            nthreads = atoi(argv[2]);
            argv += 2;
        }
        break;
    }

    if( (f = fopen(argv[1], "r")) == NULL) {
        perror("Error opening file");
        return -1;
    }

    if( (defs = slack_parse_inp(&e, f)) == NULL) {
        printf("bad parse.\n");
    } else if(e.debuglevel) {
        n = smap_iter(defs, show_assign, NULL);
        printf("Total assignments = %d\n", n);
    }

    // execute dag
    if( (a = smap_get(defs, "result")) != NULL) {
        Tensor *t;
        MemSpace *mem = memspace_ctor(32, 1<<30); // 1 Gb
        //MemSpace *mem = memspace_ctor(32, 35976/2);
        if(mem == NULL) {
            printf("Error constructing memspace.\n");
            return 1;
        }
        for(n=0; n<10; n++) {
          if( (t = run_quark(a, nthreads, mem, defs)) == NULL) {
            printf("Error executing dag.\n");
            return 1;
          }
          if(e.debuglevel) {
            printf("Result = \n");
            print_tens(stdout, t);
          }
          printf("Used mem = %lu\n", mem->used);
          tensor_dtor(&t, mem);
        }
        memspace_dtor(&mem);
    }

    smap_dtor(&defs);
 
    return 0;
}

