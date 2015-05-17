#include <cstdio>
#include <map>

extern "C" {
    #include <ast.h>
}
#include <tbb/flow_graph.h>
#include "memspace.h"

class AstNode;
class AstGraph;

class AstNode {
  public:
    const Ast *op;
    void *out; // block in graph's memory allocator object
                // there is a copy danger here...
    AstNode(const Ast *a, AstGraph *g, SMap *m); // : op(a);
    //AstNode(a) op(a.op), mem(a.mem), inps(); // copying my object is a total show-stopper
    ~AstNode() {};
    void operator()(tbb::flow::continue_msg);
    std::vector<AstNode *> inps; // node labels for inputs

  private:
    MemSpace *mem;
    AstNode *add_node(Ast *b, AstGraph *g, SMap *map);
};

typedef std::map<std::string, AstNode *> NodeMap;

typedef std::map< AstNode *, tbb::flow::continue_node<tbb::flow::continue_msg> > GMap;

// Syntax-tree graph
class AstGraph {
    public:
        AstGraph(Ast *a) { // tree-type
            head = new AstNode(a, this, NULL);
            GMap cm;
            link(head, cm);
        };
        AstGraph(Ast *a, SMap *map) { // dag-type
            head = new AstNode(a, this, map);
            GMap cm;
            link(head, cm);
        };
        ~AstGraph() {
            free(head); // TODO: traverse + mark graph...
        };

        void *run(); // run Ast

        NodeMap named; // Caution! only the head-node has a reliable node->out
        AstNode *head; // value after run() -- due to mem. reuse.
        tbb::flow::broadcast_node< tbb::flow::continue_msg > start;
        MemSpace mem;

    private:
        tbb::flow::graph g;
        tbb::flow::continue_node<tbb::flow::continue_msg>
                link(AstNode *a, GMap &cm);
};

