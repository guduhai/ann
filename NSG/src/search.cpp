// CPU search on the frozen official graph; no full-precision reranking.
// Distances are supplied by ann_quantization; graph traversal is unchanged.
#include "native_search_scorer.hpp"
#include <efanna2e/distance.h>
#include <efanna2e/util.h>
#include <efanna2e/neighbor.h>

#include <boost/dynamic_bitset.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr uint32_t D = 128, K = 10;

uint32_t parse_uint(const char *value, const char *label) {
    const std::string text(value);
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error(std::string("Invalid ") + label);
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
        if (graph_file_.size() % 4 || graph_file_.size() < 12)
            throw std::runtime_error("Invalid NSG graph size");
        words_ = reinterpret_cast<const uint32_t *>(graph_file_.data());
        const size_t word_count = graph_file_.size() / 4;
        max_degree_ = words_[0];
        entry_ = words_[1];
        size_t scan = 2;
        count_ = 0;
        while (scan < word_count) {
            const uint32_t degree = words_[scan++];
            if (degree > max_degree_ || scan + degree > word_count)
                throw std::runtime_error("Invalid graph record");
            scan += degree;
            if (count_ == UINT32_MAX) throw std::runtime_error("Too many graph nodes");
            ++count_;
        }
        if (scan != word_count || count_ < K || width_ < K || width_ > count_ || width_ >= INT32_MAX)
            throw std::runtime_error("Invalid graph node count or W");
        if (!max_degree_ || max_degree_ > count_ || entry_ >= count_)
            throw std::runtime_error("Invalid NSG graph header");
        offsets_.resize(count_);
        degrees_.resize(count_);
        size_t cursor = 2;
        for (uint32_t node = 0; node < count_; ++node) {
            if (cursor >= word_count) throw std::runtime_error("Short NSG graph");
            uint32_t degree = words_[cursor++];
            if (degree > max_degree_ || cursor + degree > word_count)
                throw std::runtime_error("Invalid NSG degree");
            offsets_[node] = cursor;
            degrees_[node] = degree;
            for (uint32_t j = 0; j < degree; ++j)
                if (words_[cursor + j] >= count_)
                    throw std::runtime_error("NSG edge outside base range");
            cursor += degree;
        }
        if (cursor != word_count) throw std::runtime_error("Trailing NSG graph data");
        if (kind == "exact") {
            base_file_ = std::make_unique<ann_quantization::MappedFile>(base_path);
            if (base_file_->size() != static_cast<size_t>(count_) * (D + 1) * 4)
                throw std::runtime_error("Base count does not match graph");
            base_ = reinterpret_cast<const float *>(base_file_->data());
            for (uint32_t i = 0; i < count_; ++i)
                if (reinterpret_cast<const uint32_t *>(base_)[static_cast<size_t>(i) * (D + 1)] != D)
                    throw std::runtime_error("Invalid fvecs dimension");
            norms_.resize(count_);
            for (uint32_t i = 0; i < count_; ++i)
                norms_[i] = distance_.norm(vector(i), D);
        } else {
            scorer_ = std::make_unique<ann_quantization::SearchScorer>(kind, prepared, model, count_, code_size);
        }
    }

    uint32_t count() const { return count_; }

    std::vector<uint32_t> operator()(const float *query) {
        if (scorer_) scorer_->prepare(query);
        auto score = [&](uint32_t id) {
            return scorer_ ? scorer_->score(id)
                           : distance_.compare(vector(id), query, norms_[id], D);
        };

        std::vector<efanna2e::Neighbor> pool(width_ + 1);
        std::vector<uint32_t> initial(width_);
        boost::dynamic_bitset<> visited(count_, 0);
        uint32_t size = 0;
        const uint32_t entry_degree = degrees_[entry_];
        for (; size < width_ && size < entry_degree; ++size) {
            initial[size] = words_[offsets_[entry_] + size];
            visited[initial[size]] = true;
        }
        while (size < width_) {
            const uint32_t id = static_cast<uint32_t>(std::rand()) % count_;
            if (visited[id]) continue;
            visited[id] = true;
            initial[size++] = id;
        }
        for (uint32_t i = 0; i < width_; ++i) {
            uint32_t id = initial[i];
            pool[i] = efanna2e::Neighbor(id, score(id), true);
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
    uint32_t width_, count_ = 0, entry_ = 0, max_degree_ = 0;
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
        if (!count || !repeats || warmup > count)
            throw std::runtime_error("Invalid query parameters");
        ann_quantization::MappedFile query_file(argv[3]);
        const size_t record_bytes = (D + 1) * sizeof(float);
        if (query_file.size() % record_bytes || query_file.size() / record_bytes > UINT32_MAX ||
            query_file.size() / record_bytes < count)
            throw std::runtime_error("Invalid query file size or query count");
        const auto *query_words = reinterpret_cast<const uint32_t *>(query_file.data());
        for (size_t i = 0; i < query_file.size() / record_bytes; ++i)
            if (query_words[i * (D + 1)] != D)
                throw std::runtime_error("Invalid query dimension");
        const float *queries = reinterpret_cast<const float *>(query_file.data());
        Search search(argv[1], argv[2], argv[4], argv[5], argv[6], width, parse_uint(argv[12], "code size"));

        std::srand(1);
        for (uint32_t i = 0; i < warmup; ++i)
            search(queries + static_cast<size_t>(i) * (D + 1) + 1);
        std::vector<uint32_t> first_ids;
        first_ids.reserve(static_cast<size_t>(count) * K);
        std::vector<double> qpm;
        for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
            std::srand(1);
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
                  << (std::string(argv[4]) == "exact" ? static_cast<uint64_t>(search.count()) * D * 4 : 0)
                  << ",\"code_size\":" << argv[12] << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

