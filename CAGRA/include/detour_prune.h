#pragma once
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <vector>
// CAGRA-style ranking on NSG's candidate set: count A->D->B where D
// precedes B in A's distance-sorted candidates. Ties: distance, then ID.
// This adapts the rule; it is not the complete cuVS GPU graph optimizer.
template<class Node, class Adjacency>
std::vector<Node> detour_prune(unsigned owner, std::vector<Node> pool,
                               unsigned degree, unsigned cap, Adjacency neighbors) {
    std::sort(pool.begin(), pool.end(), [](const Node& a,const Node& b) {
        return a.distance != b.distance ? a.distance < b.distance : a.id < b.id;
    });
    std::unordered_map<unsigned,size_t> pos;
    std::vector<Node> candidates;
    for (auto p: pool) {
        if (candidates.size() >= cap) break;
        if (p.id != owner && pos.emplace(p.id,candidates.size()).second) candidates.push_back(p);
    }
    std::vector<unsigned> count(candidates.size());
    for (size_t i=0;i<candidates.size();++i)
        for (unsigned b: neighbors(candidates[i].id)) {
            auto it=pos.find(b);
            if(it!=pos.end() && it->second>i) ++count[it->second];
        }
    std::vector<size_t> order(candidates.size());
    std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return count[a]<count[b];});
    std::vector<Node> result;
    for(size_t i=0;i<std::min<size_t>(degree,order.size());++i) result.push_back(candidates[order[i]]);
    return result;
}
