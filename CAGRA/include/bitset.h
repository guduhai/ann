#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>
namespace ann {
class Bitset {
    std::vector<uint64_t> words;
public:
    explicit Bitset(size_t n, unsigned = 0) : words((n+63)/64, 0) {}
    struct Ref {
        uint64_t& word; uint64_t mask;
        operator bool() const { return (word & mask) != 0; }
        Ref& operator=(bool v) { if(v) word |= mask; else word &= ~mask; return *this; }
    };
    Ref operator[](size_t n) { return {words[n/64], uint64_t(1)<<(n%64)}; }
    bool operator[](size_t n) const { return (words[n/64]>>(n%64))&1; }
    void reset() { std::fill(words.begin(), words.end(), 0); }
};
}
