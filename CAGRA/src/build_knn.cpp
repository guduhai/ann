#include "io.h"
#include <efanna2e/index_graph.h>
#include <efanna2e/index_random.h>
#include <omp.h>
#include <iostream>
#include <chrono>
int main(int argc,char** argv) { try {
    need(argc==8,"Usage: ann_knn base.f32 output.knn K L iterations S reverse_R");
    new_output(argv[2]);auto x=read_matrix(argv[1]);
    unsigned k=number(argv[3]),l=number(argv[4]),iter=number(argv[5]),s=number(argv[6]),r=number(argv[7]);
    need(x.n>100 && k<=l && l<x.n && s>=2 && s<=l && 2*s<x.n,"Invalid NNDescent parameters (N>100,K<=L<N,2<=S<=L,2*S<N)");
    omp_set_dynamic(0);omp_set_num_threads(1);std::srand(20261014);
    efanna2e::IndexRandom init(x.d,x.n);
    efanna2e::IndexGraph graph(x.d,x.n,efanna2e::L2,&init);
    efanna2e::Parameters p;p.Set<unsigned>("K",k);p.Set<unsigned>("L",l);
    p.Set<unsigned>("iter",iter);p.Set<unsigned>("S",s);p.Set<unsigned>("R",r);
    auto start=std::chrono::steady_clock::now();graph.Build(x.n,x.data.data(),p);
    graph.Save(argv[2]);need(std::filesystem::file_size(argv[2])==uint64_t(x.n)*(k+1)*4,"Invalid saved kNN graph");
    std::cout<<"NNDescent seconds: "<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<'\n';
    return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;} }
