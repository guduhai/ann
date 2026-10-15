// CPU search on the frozen official graph; no full-precision reranking.
// Distances are supplied by ann_quantization; graph traversal is unchanged.
#include "native_search_scorer.hpp"
#include <distance.h>
#include <util.h>
#include <neighbor.h>

#include <boost/dynamic_bitset.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr uint32_t D = 128, K = 10;

uint32_t parse_uint(const char *value, const char *label) {
    unsigned long number = std::stoul(value);
    if (number > UINT32_MAX) throw std::runtime_error(std::string("Invalid ") + label);
    return static_cast<uint32_t>(number);
}

class Search {
 public:
    Search(const std::string &graph_path, const std::string &base_path,
           const std::string &kind, const std::string &prepared,
           const std::string &model, uint32_t width, uint32_t code_size)
        : graph_file_(graph_path), width_(width) {
        count_ = 1'000'000;
        if (width_ < K || width_ > count_)
            throw std::runtime_error("Invalid W");
        if (kind == "exact") {
            base_file_ = std::make_unique<ann_quantization::MappedFile>(base_path);
            if (base_file_->size() != static_cast<size_t>(count_) * (D + 1) * 4)
                throw std::runtime_error("Expected SIFT1M base");
            base_ = reinterpret_cast<const float *>(base_file_->data());
            for (uint32_t i = 0; i < count_; ++i)
                if (reinterpret_cast<const uint32_t *>(base_)[static_cast<size_t>(i) * (D + 1)] != D)
                    throw std::runtime_error("Invalid fvecs dimension");
        }
        if (graph_file_.size() % 4 || graph_file_.size() < 12)
            throw std::runtime_error("Invalid SSG graph size");
        words_ = reinterpret_cast<const uint32_t *>(graph_file_.data());
        const size_t word_count = graph_file_.size() / 4;
        max_degree_ = words_[0];
        uint32_t entry_count = words_[1];
        if (!max_degree_ || max_degree_ > count_ || !entry_count ||
            entry_count >= count_ || 2ULL + entry_count > word_count)
            throw std::runtime_error("Invalid SSG graph header");
        entries_.assign(words_ + 2, words_ + 2 + entry_count);
        for (uint32_t entry : entries_)
            if (entry >= count_) throw std::runtime_error("SSG entry outside base range");
        offsets_.resize(count_);
        degrees_.resize(count_);
        size_t cursor = 2 + entry_count;
        for (uint32_t node = 0; node < count_; ++node) {
            if (cursor >= word_count) throw std::runtime_error("Short SSG graph");
            uint32_t degree = words_[cursor++];
            if (degree > max_degree_ || cursor + degree > word_count)
                throw std::runtime_error("Invalid SSG degree");
            offsets_[node] = cursor;
            degrees_[node] = degree;
            for (uint32_t j = 0; j < degree; ++j)
                if (words_[cursor + j] >= count_)
                    throw std::runtime_error("SSG edge outside base range");
            cursor += degree;
        }
        if (cursor != word_count) throw std::runtime_error("Trailing SSG graph data");
        if (kind == "exact") {
            norms_.resize(count_);
            for (uint32_t i = 0; i < count_; ++i)
                norms_[i] = distance_.norm(vector(i), D);
        } else {
            scorer_ = std::make_unique<ann_quantization::SearchScorer>(kind, prepared, model, count_, code_size);
        }
    }

    uint32_t count() const { return count_; }

    std::vector<uint32_t> operator()(const float *query) {
        if (entries_.size() >= width_)
            throw std::runtime_error("SSG requires W greater than entry count");
        if (scorer_) scorer_->prepare(query);
        auto score = [&](uint32_t id) {
            return scorer_ ? scorer_->score(id)
                           : distance_.compare(vector(id), query, norms_[id], D);
        };

        std::vector<efanna2e::Neighbor> pool(width_ + 1);
        std::vector<uint32_t> initial(width_);
        boost::dynamic_bitset<> visited(count_, 0);
        std::mt19937 rng(std::rand());
        efanna2e::GenRandom(rng, initial.data(), width_, count_);
        for (size_t i = 0; i < entries_.size(); ++i) initial[i] = entries_[i];
        for (uint32_t i = 0; i < width_; ++i) {
            uint32_t id = initial[i];
            pool[i] = efanna2e::Neighbor(id, score(id), true);
            visited[id] = true;
        }
        std::sort(pool.begin(), pool.begin() + width_);

        int position = 0;
        while (position < static_cast<int>(width_)) {
            int updated = static_cast<int>(width_);
            if (pool[position].flag) {
                pool[position].flag = false;
                const uint32_t node = pool[position].id;
                for (uint32_t j = 0; j < degrees_[node]; ++j) {
                    uint32_t id = words_[offsets_[node] + j];
                    if (visited[id]) continue;
                    visited[id] = true;
                    float value = score(id);
                    if (value >= pool[width_ - 1].distance) continue;
                    int at = efanna2e::InsertIntoPool(
                        pool.data(), width_, efanna2e::Neighbor(id, value, true));
                    updated = std::min(updated, at);
                }
            }
            position = updated <= position ? updated : position + 1;
        }

        std::vector<uint32_t> result;
        result.reserve(K);
        for (uint32_t i = 0; i < K; ++i) result.push_back(pool[i].id);
        return result;
    }

 private:
    const float *vector(uint32_t id) const {
        return base_ + static_cast<size_t>(id) * (D + 1) + 1;
    }

    ann_quantization::MappedFile graph_file_;
    std::unique_ptr<ann_quantization::MappedFile> base_file_;
    const uint32_t *words_ = nullptr;
    const float *base_ = nullptr;
    uint32_t width_, count_ = 0, max_degree_ = 0;
    std::vector<uint32_t> entries_;
    std::vector<size_t> offsets_;
    std::vector<uint32_t> degrees_;
    std::vector<float> norms_;
    efanna2e::DistanceFastL2 distance_;
    std::unique_ptr<ann_quantization::SearchScorer> scorer_;
};

void save_ivecs(const std::string &path, const std::vector<uint32_t> &ids) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open output ivecs");
    const uint32_t k = K;
    for (size_t i = 0; i < ids.size(); i += K) {
        out.write(reinterpret_cast<const char *>(&k), 4);
        out.write(reinterpret_cast<const char *>(ids.data() + i), K * 4);
    }
    if (!out) throw std::runtime_error("Cannot finish output ivecs");
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 13)
            throw std::runtime_error("Usage: official_quant_search GRAPH BASE QUERY KIND PREPARED MODEL W COUNT WARMUP REPEATS OUTPUT CODE_SIZE");
        uint32_t width = parse_uint(argv[7], "W");
        uint32_t count = parse_uint(argv[8], "query count");
        uint32_t warmup = parse_uint(argv[9], "warmup");
        uint32_t repeats = parse_uint(argv[10], "repeats");
        if (!count || count > 10'000 || !repeats || warmup > 10'000)
            throw std::runtime_error("Invalid query parameters");
        ann_quantization::MappedFile query_file(argv[3]);
        if (query_file.size() < static_cast<size_t>(count) * (D + 1) * 4)
            throw std::runtime_error("Too few SIFT queries");
        const float *queries = reinterpret_cast<const float *>(query_file.data());
        Search search(argv[1], argv[2], argv[4], argv[5], argv[6], width, parse_uint(argv[12], "code size"));
        if (search.count() != 10'000 && search.count() != 1'000'000)
            throw std::runtime_error("Unexpected base count");

        auto do_warmup = [&]() {
            // The author's CLI warms queries 0..9 three times.
            for (uint32_t loop = 0; loop < 3 && warmup >= 30; ++loop)
                for (uint32_t i = 0; i < 10; ++i)
                    search(queries + static_cast<size_t>(i) * (D + 1) + 1);
            for (uint32_t i = warmup >= 30 ? 30 : 0; i < warmup; ++i)
                search(queries + static_cast<size_t>(i) * (D + 1) + 1);
        };
        std::vector<uint32_t> first_ids;
        first_ids.reserve(static_cast<size_t>(count) * K);
        std::vector<double> qpm;
        for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
            std::srand(161803398);
            do_warmup();
            auto start = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < count; ++i) {
                auto ids = search(queries + static_cast<size_t>(i) * (D + 1) + 1);
                if (repeat == 0) first_ids.insert(first_ids.end(), ids.begin(), ids.end());
            }
            double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            qpm.push_back(60.0 * count / seconds);
        }
        save_ivecs(argv[11], first_ids);
        auto sorted = qpm;
        std::sort(sorted.begin(), sorted.end());
        std::cout << std::fixed << std::setprecision(3)
                  << "{\"kind\":\"" << argv[4] << "\",\"W\":" << width
                  << ",\"queries\":" << count << ",\"warmup\":" << warmup
                  << ",\"repeats\":" << repeats << ",\"median_qpm\":"
                  << sorted[sorted.size() / 2] << ",\"qpm\":[";
        for (size_t i = 0; i < qpm.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << qpm[i];
        }
        std::cout << "],\"exact_rerank\":false,\"database_float_bytes_loaded\":"
                  << (std::string(argv[4]) == "exact" ? 512000000 : 0)
                  << ",\"code_size\":" << argv[12] << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

