// Copyright David M. Rogers
// Released into Public Domain
#include <cstdio>
#include <stdlib.h>

#include <tbb/flow_graph.h>
#include "memspace.h"

using namespace tbb::flow;

extern "C" {
    #include <ast.h>
}

// TODO: pass pointers to actual data in tbb dag,
// and put a mutex on block alloc / dtor fns.
class body {
  public:
    const Ast *op;
    body( const Ast *a) : op(a) {}
    void operator()(continue_msg);
};

void body::operator()(continue_msg) {
    write_ast_label(stdout, (Ast *)op);
    printf("\n");
}

int print_v(int v) {
    printf("%d\n", v);
    sleep(1);
    printf("%d\n", v);
    return v;
}

int main() {
    graph g;
    unsigned char perm[2] = {1,0};
    Slice ind = slice_ctor(1, 2, 2);
    unsigned char *x = (unsigned char *)ind->x;
    x[0] = 1;
    x[1] = 1;

    //broadcast_node< continue_msg > start;
    broadcast_node< int > start;
    function_node< int, int > a( g, unlimited, print_v);

    /*continue_node<continue_msg> a( g, body(mkRef("Qvv")));
    continue_node<continue_msg> b( g, body(mkRef("Qov")));
    continue_node<continue_msg> c( g, body(mkScale(NULL, 2.0)));
    continue_node<continue_msg> d( g, body(mkTranspose(NULL, 2, perm)));
    continue_node<continue_msg> e( g, body(mkDot(NULL, NULL, ind)));
    */
 
    make_edge( start, a );
    /*make_edge( start, b );
    make_edge( a, c );
    make_edge( b, c );
    make_edge( c, d );
    make_edge( a, e );*/
 
    for (int i = 0; i < 3; ++i ) {
        //start.try_put( continue_msg() );
        start.try_put(31337);
        g.wait_for_all();
    }
 
    return 0;
}

