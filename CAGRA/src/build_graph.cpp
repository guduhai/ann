#include "io.h"
#include <efanna2e/index_nsg.h>
#include <omp.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <unordered_set>

int main(int argc, char** argv) { try {
    need(argc>=7 && argc<=9,
         "Usage: ann_build base.fvecs input.knn output.graph R L C [N=0] [threads=4]");
    new_output(argv[3]);
    auto x=read_vectors(argv[1],argc>=8 ? count_number(argv[7]) : 0);
    unsigned r=number(argv[4]), l=number(argv[5]), c=number(argv[6]);
    unsigned threads=argc==9 ? number(argv[8]) : 4;
    need(x.d==128,"Expected 128D vectors");
    need(r<x.n && l<x.n && l>=r && c>=r,"Require R<=L<N and C>=R");
    std::ifstream knn(argv[2],std::ios::binary); need(bool(knn),"Cannot open kNN graph");
    unsigned k=read_one<uint32_t>(knn); need(k>0 && k<x.n,"Invalid K"); knn.seekg(0);
    need(std::filesystem::file_size(argv[2])==uint64_t(x.n)*(k+1)*4,"kNN node count mismatch");
    for(unsigned i=0;i<x.n;++i) {
        need(read_one<uint32_t>(knn)==k,"Nonuniform kNN graph");
        std::unordered_set<unsigned> ids;
        for(unsigned j=0;j<k;++j) {
            auto id=read_one<uint32_t>(knn);
            need(id<x.n && id!=i && ids.insert(id).second,"Invalid/duplicate/self kNN edge");
        }
    }
    omp_set_dynamic(0); omp_set_num_threads(threads); std::srand(20261014);
    efanna2e::IndexNSG graph(x.d,x.n,efanna2e::L2,nullptr);
    efanna2e::Parameters p; p.Set<std::string>("nn_graph_path",argv[2]);
    p.Set<unsigned>("R",r); p.Set<unsigned>("L",l); p.Set<unsigned>("C",c);
    auto start=std::chrono::steady_clock::now();
    graph.Build(x.n,x.data.data(),p); graph.Save(argv[3]);
    need(std::filesystem::file_size(argv[3])>8,"Failed to save graph");
    std::cout<<"Configured build threads: "<<omp_get_max_threads()<<"\nGraph build seconds: "
             <<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<'\n';
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; } }
