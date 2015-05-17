// Copyright David M. Rogers

#include <stdlib.h>

#include "ast_graph.h"

using namespace tbb::flow;

AstNode::AstNode(const Ast *a, AstGraph *g, SMap *map) : op(a), mem(&g->mem) {
    Ast **t;
    const Ast *orig;
    int m = ast_children((Ast *)a, &t);

    for(int i=0; i<m; i++) {
        inps.push_back(add_node(t[i], g, map));
    }
}

// Helper function to traverse named links correctly.
AstNode *AstNode::add_node(Ast *b, AstGraph *g, SMap *map) {
    Ast *orig = b;
    AstNode *c;
    NodeMap::iterator r;

    while(b->type == TRef) {
        std::string name(b->ref);

        if( (r = g->named.find(name)) == g->named.end())
            break; // done

        if(map == NULL || (b = (Ast *)smap_get(map, b->ref)) == NULL) {
            printf("Undeclared var: %s\n", b->ref);
            exit(1);
        }
    }

    if(b->type != TRef) {
        c = new AstNode(b, g, map);
    } else {
        c = r->second;
    }

    // Re-traverse to store names in g->named.
    while(orig != b) {
        std::string name(orig->ref);
        g->named[name] = c;
    }
    return c;
}

void AstNode::operator()(tbb::flow::continue_msg) {
    int i;
    void **inp = (void **)malloc(inps.size()*sizeof(void *));
    for(i=0; i<inps.size(); i++) {
        inp[i] = inps[i]->out;
    }
    out = exec_ast(op, inps.size(), inp, (void *)mem);
    for(i=0; i<inps.size(); i++) {
        release(inp[i]);
    }
    free(inp);
}

void *AstGraph::run() {
    start.try_put(continue_msg());
    g.wait_for_all();
}


// In a better framework, this would have been done during
// AstNode construction.  Here, node body-s are not supposed
// to know about the graph!!!?!?!??
//
continue_node<continue_msg> AstGraph::link(AstNode *a, GMap &cm) {
    int i;
    GMap::iterator r;

    if( (r = cm.find(a)) == cm.end()) {
        return r->second;
    }

    continue_node<continue_msg> self(g, *a);

    for(i=0; i<a->inps.size(); i++) {
        continue_node<continue_msg>child = link(a->inps[i], cm);
        // error: invalid initialization of non-const reference
        // due to poor design of make_edge, when continue_nodes
        // are _supposed_ to be copied on input, etc.
        make_edge(child, self);
    }
    if(i == 0) { // terminal values are activated by start...
        make_edge(start, self);
    }

    return self;
}

