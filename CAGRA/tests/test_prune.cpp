#include "detour_prune.h"
#include <stdexcept>
struct Node {unsigned id;float distance;};
int main(){
    // 0->1->2 and 0->1->3 are detours. Edge 0->4 has none even
    // though it is farther away; it must beat 2 and 3 at R=2.
    std::vector<std::vector<unsigned>> g={{1,2,3,4},{2,3},{3},{},{}};
    auto row=[&](unsigned i)->const std::vector<unsigned>&{return g.at(i);};
    auto selected=detour_prune(0,std::vector<Node>{{3,3},{1,1},{2,2},{4,4},{0,0},{1,1}},2,10,row);
    if(selected.size()!=2 || selected[0].id!=1 || selected[1].id!=4) throw std::runtime_error("detour count/rank failed");
    selected=detour_prune(0,std::vector<Node>{{0,0}},2,10,row);
    if(!selected.empty()) throw std::runtime_error("self edge retained");
    return 0;
}
