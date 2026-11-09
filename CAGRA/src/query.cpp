#include "io.h"
#include "native_search_scorer.hpp"
#ifndef ANN_DIM
#define ANN_DIM 128
#endif
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>

// Shared Linux/Android query: native NSG adjacency, no CSR conversion.
// Native adjacency is mapped read-only, as in the NSG/NSSG adapters.
struct Graph {
    ann_quantization::MappedFile file;
    uint32_t n, entry;
    const uint32_t* words;
    std::vector<size_t> offsets;
    std::vector<uint32_t> degrees;
    Graph(const std::string& path,uint32_t count):file(path),n(count),offsets(n),degrees(n) {
        need(file.size()>=12 && file.size()%4==0,"Invalid native adjacency size");
        words=reinterpret_cast<const uint32_t*>(file.data());
        uint32_t width=words[0]; entry=words[1];
        need(width>0 && width<n && entry<n,"Invalid native adjacency header");
        size_t cursor=2, length=file.size()/4;
        for(uint32_t i=0;i<n;++i) {
            need(cursor<length,"Truncated graph");
            uint32_t k=words[cursor++];
            need(k>0 && k<=width && cursor+k<=length,"Invalid degree");
            offsets[i]=cursor; degrees[i]=k;
            for(uint32_t j=0;j<k;++j) need(words[cursor+j]<n && words[cursor+j]!=i,"Invalid edge");
            cursor+=k;
        }
        need(cursor==length,"Graph node count mismatch or trailing data");
    }
    const uint32_t* begin(uint32_t id) const { return words+offsets[id]; }
    const uint32_t* end(uint32_t id) const { return begin(id)+degrees[id]; }
};
struct Cand { float dist; uint32_t id; bool fresh; };
static bool nearer(const Cand& a,const Cand& b) {
    return a.dist!=b.dist ? a.dist<b.dist : a.id<b.id;
}
static std::vector<uint32_t> read_truth(const std::string& path,uint32_t q,uint32_t n) {
    std::ifstream f(path,std::ios::binary); need(bool(f),"Cannot open truth");
    char header[8]{}; need(bool(f.read(header,8)),"Truncated truth");
    bool matrix=std::string(header,8)=="EGI32001";
    uint32_t columns=0;
    if(matrix) {
        uint32_t rows=read_one<uint32_t>(f); columns=read_one<uint32_t>(f);
        need(rows>=q && columns>=10 && columns<=65536,"Invalid truth matrix shape");
        need(std::filesystem::file_size(path)==16+uint64_t(rows)*columns*4,"Wrong truth size");
    } else { f.clear(); f.seekg(0); }
    std::vector<uint32_t> truth(size_t(q)*10);
    for(uint32_t i=0;i<q;++i) {
        uint32_t k=matrix ? columns : read_one<uint32_t>(f);
        need(k>=10 && k<=65536,"Invalid truth row");
        auto row=read_array<uint32_t>(f,k);
        for(unsigned j=0;j<10;++j) need(row[j]<n,"Truth ID exceeds graph size; subset requires new ground truth");
        std::copy_n(row.begin(),10,truth.begin()+size_t(i)*10);
    }
    return truth;
}
static double peak() {
    std::ifstream f("/proc/self/status"); std::string s;
    while(std::getline(f,s)) if(s.rfind("VmHWM:",0)==0) {
        std::istringstream in(s.substr(6)); double x;
        need(bool(in>>x),"Bad VmHWM"); return x/1024.;
    }
    throw std::runtime_error("VmHWM missing");
}
int main(int argc,char** argv) { try {
    need(argc==15,"Usage: ann_query GRAPH BASE QUERY TRUTH N Q KIND CODES MODEL L WARMUP REPEATS PREFIX CODE_SIZE");
    uint32_t n=number(argv[5]), qcount=number(argv[6]), L=number(argv[10]);
    uint32_t warm=count_number(argv[11]), reps=number(argv[12]), size=count_number(argv[14]);
    std::string kind=argv[7], prefix=argv[13];
    need(kind=="float32" || kind=="int8" || kind=="pca" || kind=="pq" || kind=="local","Unknown method");
    need(warm<=qcount && reps<=100 && L>=10 && L<=n,"Invalid query parameters");
    new_output(prefix+".json"); new_output(prefix+".labels");
    auto start=std::chrono::steady_clock::now();
    Graph g(argv[1],n);
    Matrix base{};
    // Quantized modes never open argv[2], and never rerank using float vectors.
    if(kind=="float32") { base=read_vectors(argv[2],n); need(base.d==ANN_DIM,"Base dimension does not match query executable"); }
    auto queries=read_vectors(argv[3],qcount); need(queries.d==ANN_DIM,"Query dimension does not match executable");
    auto truth=read_truth(argv[4],qcount,n);
    std::unique_ptr<ann_quantization::SearchScorer> scorer;
    if(kind!="float32") scorer=std::make_unique<ann_quantization::SearchScorer>(kind,argv[8],argv[9],n,size);
    auto score=[&](uint32_t id,const float* q) {
        if(scorer) return scorer->score(id);
        const float* x=base.data.data()+size_t(id)*ANN_DIM; float sum=0;
        for(int j=0;j<ANN_DIM;++j) { float d=x[j]-q[j]; sum+=d*d; }
        return sum;
    };
    // Preserve the Pi traversal, including reusable epoch workspace and entry-relative fill.
    std::vector<uint32_t> seen(n); uint32_t epoch=0;
    std::vector<Cand> pool; pool.reserve(L+1);
    auto one=[&](uint32_t qi,uint32_t* out) {
        const float* q=queries.data.data()+size_t(qi)*ANN_DIM;
        if(scorer) scorer->prepare(q);
        if(++epoch==0) { std::fill(seen.begin(),seen.end(),0); ++epoch; }
        pool.clear();
        auto add=[&](uint32_t id) {
            if(seen[id]!=epoch) { seen[id]=epoch; pool.push_back({score(id,q),id,true}); }
        };
        add(g.entry);
        for(auto ptr=g.begin(g.entry);ptr!=g.end(g.entry) && pool.size()<L;++ptr) add(*ptr);
        for(uint32_t i=0;pool.size()<L && i<n;++i) add(uint32_t((uint64_t(g.entry)+i)%n));
        std::sort(pool.begin(),pool.end(),nearer);
        int pos=0;
        while(pos<int(L)) {
            int updated=L; Cand cur=pool[pos];
            if(cur.fresh) {
                pool[pos].fresh=false;
                for(auto ptr=g.begin(cur.id);ptr!=g.end(cur.id);++ptr) {
                    auto id=*ptr;
                    if(seen[id]==epoch) continue;
                    seen[id]=epoch; Cand c{score(id,q),id,true};
                    if(!nearer(c,pool.back())) continue;
                    auto it=std::lower_bound(pool.begin(),pool.end(),c,nearer);
                    int at=int(it-pool.begin()); pool.insert(it,c); pool.pop_back();
                    updated=std::min(updated,at);
                }
            }
            pos=updated<=pos ? updated : pos+1;
        }
        for(int k=0;k<10;++k) out[k]=pool[k].id;
    };
    std::vector<uint32_t> warmout(10), labels(size_t(qcount)*10);
    double load_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    for(uint32_t i=0;i<warm;++i) one(i,warmout.data());
    std::vector<double> times;
    for(uint32_t r=0;r<reps;++r) {
        auto t=std::chrono::steady_clock::now();
        for(uint32_t i=0;i<qcount;++i) one(i,labels.data()+size_t(i)*10);
        times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count());
    }
    auto ordered=times; std::sort(ordered.begin(),ordered.end());
    double sec=ordered.size()%2 ? ordered[ordered.size()/2]
        : (ordered[ordered.size()/2-1]+ordered[ordered.size()/2])/2;
    size_t hits=0;
    for(uint32_t i=0;i<qcount;++i) for(int a=0;a<10;++a) for(int b=0;b<10;++b)
        if(labels[size_t(i)*10+a]==truth[size_t(i)*10+b]) { ++hits; break; }
    double recall=double(hits)/(size_t(qcount)*10), qpm=60.*qcount/sec, rss=peak();
    std::ofstream ids(prefix+".labels",std::ios::binary);
    ids.write(reinterpret_cast<char*>(labels.data()),labels.size()*4); ids.close();
    need(bool(ids),"Cannot write labels");
    std::ofstream js(prefix+".json");
    js<<std::setprecision(12)<<"{\"method\":\""<<kind<<"\",\"vectors\":"<<n
      <<",\"dimensions\":"<<ANN_DIM<<",\"queries\":"<<qcount<<",\"search_l\":"<<L<<",\"warmup\":"<<warm
      <<",\"repeats\":"<<reps<<",\"code_size\":"<<size<<",\"query_threads\":1,\"recall_at_10\":"<<recall
      <<",\"qpm\":"<<qpm<<",\"query_seconds\":"<<sec<<",\"load_seconds\":"<<load_s
      <<",\"peak_rss_kib\":"<<rss*1024<<",\"peak_rss_mib\":"<<rss<<",\"full_precision_base_loaded\":"<<(kind=="float32"?"true":"false")
      <<",\"database_float_bytes_loaded\":"<<(kind=="float32"?uint64_t(n)*ANN_DIM*4:0)<<",\"rerank\":false,\"graph_format\":\"native adjacency\",\"repeat_seconds\":[";
    for(size_t i=0;i<times.size();++i) js<<(i?",":"")<<times[i];
    js<<"]}\n"; js.close(); need(bool(js),"Cannot write metrics");
    std::cout<<kind<<" Recall@10="<<recall<<" QPM="<<qpm<<" VmHWM_MiB="<<rss<<'\n';
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; } }
