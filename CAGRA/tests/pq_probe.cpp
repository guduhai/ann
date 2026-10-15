// Diagnostic only: reuse the actual production loader/distance/search without
// modifying ann_query or enabling reranking in the production executable.
#define main production_query_main
#include "../src/query.cpp"
#undef main

int main(int argc,char** argv) {try {
    need(argc==7,"Usage: pq_probe graph pq.bin queries output_prefix L-list emit-all-scores(0/1)");
    Graph g(argv[1]);Codec codec(argv[2]);auto queries=read_matrix(argv[3]);
    need(codec.c.mode==3 && codec.c.n==int(g.n) && codec.c.d==int(g.d) && queries.d==g.d,"PQ shape mismatch");
    std::string prefix=argv[4];
    if(std::string(argv[6])=="1") {
        new_output(prefix+".scores");std::ofstream f(prefix+".scores",std::ios::binary);
        std::vector<float> scores(g.n);
        for(uint32_t i=0;i<queries.n;++i) {
            all_scores<3>(codec.c,queries.data.data()+size_t(i)*g.d,scores);
            f.write(reinterpret_cast<const char*>(scores.data()),scores.size()*4);
        }
        need(bool(f),"Cannot write scores");
    }
    std::istringstream ls(argv[5]);std::string item;
    while(std::getline(ls,item,',')) {
        unsigned L=number(item.c_str());need(L>=10 && L<=g.n,"Invalid L");
        auto path=prefix+".l"+item+".pool";new_output(path);std::ofstream f(path,std::ios::binary);
        Workspace w(g.n,L);std::vector<uint32_t> labels(10),pool(L);
        for(uint32_t i=0;i<queries.n;++i) {
            search<3>(g,codec.c,queries.data.data()+size_t(i)*g.d,1,L,w,labels);
            for(unsigned j=0;j<L;++j)pool[j]=w.pool[j].id;
            f.write(reinterpret_cast<const char*>(pool.data()),pool.size()*4);
        }
        need(bool(f),"Cannot write pool");
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
