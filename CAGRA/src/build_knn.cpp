#include "io.h"
#include <efanna2e/index_graph.h>
#include <efanna2e/index_random.h>
#include <omp.h>
#include <chrono>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) { try {
    need(argc>=8 && argc<=10,
         "Usage: ann_knn base.fvecs output.knn K L iterations S reverse_R [N=0] [threads=4]");
    new_output(argv[2]);
    auto x=read_vectors(argv[1],argc>=9 ? count_number(argv[8]) : 0);
    unsigned k=number(argv[3]), l=number(argv[4]), iter=number(argv[5]);
    unsigned s=number(argv[6]), r=number(argv[7]), threads=argc==10 ? number(argv[9]) : 4;
    need(x.d==128,"Expected 128D vectors");
    need(x.n>100 && k<=l && l<x.n && s>=2 && s<=l && 2*s<x.n,"Invalid NNDescent parameters");
    omp_set_dynamic(0); omp_set_num_threads(threads); std::srand(20261014);
    efanna2e::IndexRandom init(x.d,x.n);
    efanna2e::IndexGraph graph(x.d,x.n,efanna2e::L2,&init);
    efanna2e::Parameters p;
    p.Set<unsigned>("K",k); p.Set<unsigned>("L",l); p.Set<unsigned>("iter",iter);
    p.Set<unsigned>("S",s); p.Set<unsigned>("R",r);
    auto start=std::chrono::steady_clock::now();
    graph.Build(x.n,x.data.data(),p); graph.Save(argv[2]);
    need(std::filesystem::file_size(argv[2])==uint64_t(x.n)*(k+1)*4,"Invalid saved kNN graph");
    std::cout<<"Configured build threads: "<<omp_get_max_threads()<<"\nNNDescent seconds: "
             <<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<'\n';
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; } }
