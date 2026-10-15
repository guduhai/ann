#include "io.h"
#include "distance_kernels.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <sys/resource.h>

static uint64_t process_peak_rss() {
    // VmHWM belongs to the current exec image. ru_maxrss may include the
    // Python parent's inherited high-water mark from before exec.
    std::ifstream status("/proc/self/status");std::string line;
    while(std::getline(status,line)) if(line.rfind("VmHWM:",0)==0){
        std::istringstream s(line.substr(6));uint64_t kib;need(bool(s>>kib),"Invalid VmHWM");return kib*1024;
    }
    throw std::runtime_error("VmHWM unavailable on this Linux host");
}

struct Graph {
    uint32_t n,d,entry;std::vector<uint64_t> offsets;std::vector<uint32_t> ids;
    explicit Graph(const std::string& path) {
        std::ifstream f(path,std::ios::binary);need(bool(f),"Cannot open graph");magic(f,"EGCSR001");
        n=read_one<uint32_t>(f);d=read_one<uint32_t>(f);entry=read_one<uint32_t>(f);
        need(read_one<uint32_t>(f)==0,"Unknown graph flags");auto edges=read_one<uint64_t>(f);
        need(n>0 && n<2147483647 && entry<n && d>0 && d<=65536,"Invalid graph header");
        need(edges<=uint64_t(n)*n && edges <= (std::filesystem::file_size(path)/4),"Invalid edge count");
        need(std::filesystem::file_size(path)==32+(uint64_t(n)+1)*8+edges*4,"Invalid graph size");
        offsets=read_array<uint64_t>(f,size_t(n)+1);ids=read_array<uint32_t>(f,edges);
        need(offsets[0]==0 && offsets[n]==edges,"Invalid offsets");
        for(size_t i=0;i<n;++i) need(offsets[i]<=offsets[i+1] && offsets[i+1]<=edges,"Invalid offset order");
        for(auto id:ids) need(id<n,"Invalid neighbor ID");
    }
    size_t bytes() const {return offsets.size()*8+ids.size()*4;}
};

struct Codec {
    Config c; std::vector<float> floats,scale,mean,basis,books,centers;std::vector<uint8_t> bytes;
    explicit Codec(const std::string& path) {
        std::ifstream f(path,std::ios::binary);need(bool(f),"Cannot open codec");magic(f,"EGCODE01");
        c.n=read_one<uint32_t>(f);c.d=read_one<uint32_t>(f);c.mode=read_one<uint32_t>(f);
        c.width=read_one<uint32_t>(f);c.centroids=read_one<uint32_t>(f);c.clusters=read_one<uint32_t>(f);
        need(c.n>0 && c.d>0 && c.d<=65536 && c.mode>=0 && c.mode<=4,"Invalid codec header");
        need(c.width>=0 && c.width<=c.d && c.centroids>=0 && c.centroids<=256 && c.clusters>=0 && c.clusters<=256,"Invalid codec parameters");
        if(c.mode==2) need(c.width>0,"Invalid PCA width");
        if(c.mode==3) need(c.width>0 && c.d%c.width==0 && c.centroids>=2,"Invalid PQ shape");
        if(c.mode==4) need(c.clusters>=2,"Invalid local clusters");
        uint64_t stride=c.mode==0?c.d*4:c.mode==1?c.d:c.mode==2?c.width*4:c.mode==3?c.width:c.d+1;
        uint64_t count=read_one<uint64_t>(f);need(count==uint64_t(c.n)*stride,"Invalid code byte count");
        size_t ns=(c.mode==1?c.d:c.mode==4?c.clusters*c.d:0);
        size_t nm=c.mode==2?c.d:0,nb=c.mode==2?size_t(c.d)*c.width:0;
        size_t nk=c.mode==3?size_t(c.centroids)*c.d:0,nc=c.mode==4?c.clusters*c.d:0;
        need(std::filesystem::file_size(path)==40+count+4*(ns+nm+nb+nk+nc),"Invalid codec file size");
        if(c.mode==0 || c.mode==2) {floats=read_array<float>(f,count/4);c.codes=floats.data();}
        else {bytes=read_array<uint8_t>(f,count);c.codes=bytes.data();}
        scale=read_array<float>(f,ns);mean=read_array<float>(f,nm);basis=read_array<float>(f,nb);
        books=read_array<float>(f,nk);centers=read_array<float>(f,nc);
        c.scale=scale.data();c.mean=mean.data();c.basis=basis.data();c.codebooks=books.data();c.centers=centers.data();
        for(const auto* a:{&floats,&scale,&mean,&basis,&books,&centers})
            for(float v:*a) need(std::isfinite(v),"Nonfinite codec value");
        for(float v:scale) need(v>0,"Invalid codec scale");
        if(c.mode==3) for(auto v:bytes) need(v<c.centroids,"PQ code out of range");
        if(c.mode==4) for(int i=0;i<c.n;++i) need(bytes[size_t(i)*(c.d+1)]<c.clusters,"Cluster ID out of range");
        end_of_file(f);
    }
    size_t payload_bytes() const {return bytes.size()+4*(floats.size()+scale.size()+mean.size()+basis.size()+books.size()+centers.size());}
};

struct Candidate {float distance;uint32_t id;bool fresh;};
static bool nearer(const Candidate& a,const Candidate& b) {return a.distance!=b.distance ? a.distance<b.distance:a.id<b.id;}
struct Workspace {std::vector<uint32_t> visited;std::vector<Candidate> pool;uint32_t epoch=0;
    Workspace(size_t n,size_t l):visited(n){pool.reserve(l+1);}
};
template<int Mode> void search(const Graph& g,const Config& c,const float* queries,int nq,int L,
                               Workspace& w,std::vector<uint32_t>& labels) {
    Distance<Mode> dist(c);
    for(int qi=0;qi<nq;++qi){
        if(++w.epoch==0){std::fill(w.visited.begin(),w.visited.end(),0);++w.epoch;}
        auto& pool=w.pool;pool.clear();dist.prepare(queries+size_t(qi)*c.d);
        auto add=[&](uint32_t id){if(w.visited[id]!=w.epoch){w.visited[id]=w.epoch;pool.push_back({dist(id),id,true});}};
        add(g.entry);
        for(uint64_t j=g.offsets[g.entry];j<g.offsets[g.entry+1] && pool.size()<size_t(L);++j) add(g.ids[j]);
        // Deterministic, bounded initialization independent of libc RNG.
        for(uint32_t j=0;pool.size()<size_t(L) && j<g.n;++j) add(uint32_t((uint64_t(g.entry)+j)%g.n));
        std::sort(pool.begin(),pool.end(),nearer);
        int position=0;
        while(position<L){
            int updated=L;auto current=pool[position];
            if(current.fresh){
                pool[position].fresh=false;
                for(uint64_t j=g.offsets[current.id];j<g.offsets[current.id+1];++j){
                    auto id=g.ids[j];if(w.visited[id]==w.epoch)continue;w.visited[id]=w.epoch;
                    Candidate candidate{dist(id),id,true};if(!nearer(candidate,pool.back()))continue;
                    auto it=std::lower_bound(pool.begin(),pool.end(),candidate,nearer);int at=int(it-pool.begin());
                    pool.insert(it,candidate);pool.pop_back();updated=std::min(updated,at);
                }
            }
            position=updated<=position?updated:position+1;
        }
        // No reranking and no original float32 base-vector pointer.
        for(int k=0;k<10;++k) labels[size_t(qi)*10+k]=pool[k].id;
    }
}
static void dispatch(const Graph& g,const Config& c,const Matrix& q,int nq,int L,Workspace& w,std::vector<uint32_t>& out){
    switch(c.mode){
    case 0:search<0>(g,c,q.data.data(),nq,L,w,out);break;
    case 1:search<1>(g,c,q.data.data(),nq,L,w,out);break;
    case 2:search<2>(g,c,q.data.data(),nq,L,w,out);break;
    case 3:search<3>(g,c,q.data.data(),nq,L,w,out);break;
    case 4:search<4>(g,c,q.data.data(),nq,L,w,out);break;
    }
}
template<int Mode> void all_scores(const Config& c,const float* q,std::vector<float>& out){
    Distance<Mode> d(c);d.prepare(q);for(int i=0;i<c.n;++i)out[i]=d(i);
}
int main(int argc,char** argv){try{
    if(argc==4 && std::string(argv[1])=="scores"){
        Codec codec(argv[2]);auto q=read_matrix(argv[3]);need(q.d==unsigned(codec.c.d),"Dimension mismatch");
        std::vector<float> scores(codec.c.n);
        switch(codec.c.mode){
        case 0:all_scores<0>(codec.c,q.data.data(),scores);break;
        case 1:all_scores<1>(codec.c,q.data.data(),scores);break;
        case 2:all_scores<2>(codec.c,q.data.data(),scores);break;
        case 3:all_scores<3>(codec.c,q.data.data(),scores);break;
        case 4:all_scores<4>(codec.c,q.data.data(),scores);break;
        }
        std::cout.write(reinterpret_cast<const char*>(scores.data()),scores.size()*4);return 0;
    }
    need(argc==9,"Usage: ann_query graph codec queries truth output_prefix L warmup repeats; or: ann_query scores codec queries");
    new_output(std::string(argv[5])+".json");new_output(std::string(argv[5])+".labels");
    auto load_start=std::chrono::steady_clock::now();Graph g(argv[1]);Codec codec(argv[2]);auto q=read_matrix(argv[3]);
    need(g.n==unsigned(codec.c.n) && g.d==unsigned(codec.c.d) && q.d==g.d,"Graph/codec/query shape mismatch");
    std::ifstream tf(argv[4],std::ios::binary);magic(tf,"EGI32001");
    need(read_one<uint32_t>(tf)==q.n && read_one<uint32_t>(tf)==10,"Wrong truth shape");
    auto truth=read_array<uint32_t>(tf,size_t(q.n)*10);end_of_file(tf);for(auto id:truth)need(id<g.n,"Invalid truth ID");
    int L=number(argv[6]),warm=number(argv[7]),repeats=number(argv[8]);
    need(L>=10 && unsigned(L)<=g.n && unsigned(warm)<=q.n && repeats<=100,"Invalid query parameters");
    Workspace work(g.n,L);std::vector<uint32_t> labels(size_t(q.n)*10),reference(labels.size());
    double load_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-load_start).count();
    dispatch(g,codec.c,q,warm,L,work,labels);
    std::vector<double> seconds;
    for(int r=0;r<repeats;++r){
        auto start=std::chrono::steady_clock::now();dispatch(g,codec.c,q,q.n,L,work,labels);
        seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
        if(r==0)reference=labels;else need(reference==labels,"Non-repeatable query results");
    }
    size_t hits=0;
    for(size_t i=0;i<q.n;++i)for(int k=0;k<10;++k)
        if(std::find(truth.begin()+i*10,truth.begin()+i*10+10,labels[i*10+k])!=truth.begin()+i*10+10)++hits;
    auto sorted=seconds;std::sort(sorted.begin(),sorted.end());
    double median=sorted.size()%2?sorted[sorted.size()/2]:(sorted[sorted.size()/2-1]+sorted[sorted.size()/2])/2;
    rusage usage{};need(getrusage(RUSAGE_SELF,&usage)==0,"Cannot measure RSS");
    auto peak_rss=process_peak_rss();
    std::ofstream raw(std::string(argv[5])+".labels",std::ios::binary);
    raw.write(reinterpret_cast<const char*>(labels.data()),labels.size()*4);need(bool(raw),"Cannot write labels");
    std::ofstream out(std::string(argv[5])+".json");out<<std::setprecision(12);
    out<<"{\"mode\":"<<codec.c.mode<<",\"vectors\":"<<g.n<<",\"dimension\":"<<g.d
       <<",\"queries\":"<<q.n<<",\"search_l\":"<<L<<",\"rerank\":false,\"full_precision_base_loaded\":"<<(codec.c.mode==0?"true":"false")
       <<",\"graph_payload_bytes\":"<<g.bytes()<<",\"codec_payload_bytes\":"<<codec.payload_bytes()
       <<",\"peak_rss_bytes\":"<<peak_rss<<",\"memory_metric\":\"Linux /proc/self/status VmHWM\",\"ru_maxrss_including_pre_exec_bytes\":"<<uint64_t(usage.ru_maxrss)*1024<<",\"load_seconds\":"<<load_s
       <<",\"warmup\":"<<warm<<",\"repeats\":"<<repeats<<",\"recall_at_10\":"<<double(hits)/(q.n*10)
       <<",\"qps\":"<<q.n/median<<",\"qpm\":"<<60*q.n/median<<",\"query_seconds\":[";
    for(size_t i=0;i<seconds.size();++i){if(i)out<<',';out<<seconds[i];}out<<"]}\n";need(bool(out),"Cannot write result");
    std::cout<<"mode="<<codec.c.mode<<" L="<<L<<" recall="<<double(hits)/(q.n*10)<<" QPM="<<60*q.n/median<<" RSS_KiB="<<peak_rss/1024<<'\n';
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
