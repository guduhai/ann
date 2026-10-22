#pragma once
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <vector>
#include <stdexcept>
#include <string>
#include <cmath>
inline void need(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
template<class T>T read_one(std::istream&f){T v{};need(bool(f.read(reinterpret_cast<char*>(&v),sizeof(v))),"Truncated input");return v;}
template<class T>std::vector<T> read_array(std::istream&f,size_t n){std::vector<T>a(n);if(n)need(bool(f.read(reinterpret_cast<char*>(a.data()),n*sizeof(T))),"Truncated array");return a;}
inline void end_of_file(std::istream&f){need(f.peek()==std::char_traits<char>::eof(),"Unexpected trailing bytes");}
inline void magic(std::istream&f,const std::string&e){auto a=read_array<char>(f,8);need(std::string(a.begin(),a.end())==e,"Wrong magic/version");}
struct Matrix{uint32_t n,d;std::vector<float>data;};
inline Matrix read_matrix(const std::string&p,uint32_t limit=0){std::ifstream f(p,std::ios::binary);need(bool(f),"Cannot open matrix: "+p);magic(f,"EGF32001");Matrix m;m.n=read_one<uint32_t>(f);m.d=read_one<uint32_t>(f);need(m.n&&m.d&&m.n<2147483647&&m.d<=65536,"Invalid matrix dimensions");need(std::filesystem::file_size(p)==16+uint64_t(m.n)*m.d*4,"Wrong matrix size");if(limit){need(limit<=m.n,"Vector limit exceeds file");m.n=limit;}m.data=read_array<float>(f,size_t(m.n)*m.d);for(float v:m.data)need(std::isfinite(v),"Nonfinite vector");return m;}
inline Matrix read_vectors(const std::string&p,uint32_t limit=0){
 std::ifstream f(p,std::ios::binary);need(bool(f),"Cannot open vectors: "+p);char h[8]{};need(bool(f.read(h,8)),"Truncated vector file");
 if(std::string(h,8)=="EGF32001"){f.close();return read_matrix(p,limit);}
 f.clear();f.seekg(0);uint32_t d=read_one<uint32_t>(f);need(d&&d<=65536,"Invalid fvecs dimension");uint64_t stride=uint64_t(d+1)*4,bytes=std::filesystem::file_size(p);need(bytes%stride==0,"Invalid fvecs length");uint64_t total=bytes/stride,n=limit?limit:total;need(n&&n<=total&&n<2147483647,"Invalid fvecs row limit");Matrix m{uint32_t(n),d,std::vector<float>(size_t(n)*d)};f.seekg(0);for(uint32_t i=0;i<m.n;++i){need(read_one<uint32_t>(f)==d,"Inconsistent fvecs dimension");need(bool(f.read(reinterpret_cast<char*>(m.data.data()+size_t(i)*d),size_t(d)*4)),"Truncated fvecs row");}for(float v:m.data)need(std::isfinite(v),"Nonfinite vector");return m;
}
inline unsigned count_number(const char* s) {
 std::string a(s);need(!a.empty()&&a.find_first_not_of("0123456789")==std::string::npos,"Invalid nonnegative integer");
 size_t used;auto n=std::stoull(a,&used);need(used==a.size()&&n<2147483647,"Integer out of range");return unsigned(n);
}
inline unsigned number(const char*s){auto n=count_number(s);need(n>0,"Expected positive integer");return n;}
inline void new_output(const std::string&p){need(!std::filesystem::exists(p),"Output already exists: "+p);}
