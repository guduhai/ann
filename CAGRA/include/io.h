#pragma once
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <vector>
#include <stdexcept>
#include <string>
#include <cmath>
inline void need(bool ok,const std::string& message) { if(!ok) throw std::runtime_error(message); }
template<class T> T read_one(std::istream& f) { T v{}; need(bool(f.read(reinterpret_cast<char*>(&v),sizeof(v))),"Truncated input"); return v; }
template<class T> std::vector<T> read_array(std::istream& f,size_t n) {
    std::vector<T> a(n);
    if(n) need(bool(f.read(reinterpret_cast<char*>(a.data()),n*sizeof(T))),"Truncated array");
    return a;
}
inline void end_of_file(std::istream& f) { need(f.peek()==std::char_traits<char>::eof(),"Unexpected trailing bytes"); }
inline void magic(std::istream& f,const std::string& expected) {
    auto a=read_array<char>(f,8);need(std::string(a.begin(),a.end())==expected,"Wrong magic/version");
}
struct Matrix { uint32_t n,d; std::vector<float> data; };
inline Matrix read_matrix(const std::string& path) {
    std::ifstream f(path,std::ios::binary);need(bool(f),"Cannot open matrix: "+path);
    magic(f,"EGF32001");Matrix m; m.n=read_one<uint32_t>(f);m.d=read_one<uint32_t>(f);
    need(m.n>0 && m.d>0 && m.n<2147483647 && m.d<=65536,"Invalid matrix dimensions");
    need(std::filesystem::file_size(path)==16+uint64_t(m.n)*m.d*4,"Wrong matrix size");
    m.data=read_array<float>(f,size_t(m.n)*m.d);
    for(float v:m.data) need(std::isfinite(v),"Nonfinite vector");
    return m;
}
inline unsigned number(const char* s) {
    size_t used;std::string a(s);auto n=std::stoul(a,&used);
    need(used==a.size() && !a.empty() && a[0]!='-' && n>0 && n<2147483647,"Invalid positive integer");return n;
}
inline void new_output(const std::string& p) { need(!std::filesystem::exists(p),"Output already exists: "+p); }
