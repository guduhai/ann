// CPU search on the frozen official graph; no full-precision reranking.
// Distances are supplied by ann_quantization; graph traversal is unchanged.
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include "neighbor.h"
#include "native_search_scorer.hpp"
#include <memory>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

struct MappedFile {
    void *data = nullptr;
    size_t size = 0;
    MappedFile() = default;
    explicit MappedFile(const std::string &path) { open_file(path); }
    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;
    ~MappedFile() { if (data) munmap(data, size); }

    void open_file(const std::string &path) {
        if (data) throw std::runtime_error("File map already open");
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Cannot open " + path);
        struct stat info {};
        if (fstat(fd, &info) != 0 || info.st_size <= 0) {
            ::close(fd);
            throw std::runtime_error("Cannot stat " + path);
        }
        size = static_cast<size_t>(info.st_size);
        data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (data == MAP_FAILED) {
            data = nullptr;
            throw std::runtime_error("Cannot map " + path);
        }
    }
    const uint8_t *bytes() const { return static_cast<const uint8_t *>(data); }
};

uint32_t u32(const uint8_t *at) { uint32_t value; std::memcpy(&value, at, 4); return value; }
uint64_t u64(const uint8_t *at) { uint64_t value; std::memcpy(&value, at, 8); return value; }

struct FloatBin {
    MappedFile file;
    uint32_t count = 0, dimension = 0;
    const float *vectors = nullptr;
    explicit FloatBin(const std::string &path) : file(path) {
        if (file.size < 8) throw std::runtime_error("Short fbin: " + path);
        count = u32(file.bytes());
        dimension = u32(file.bytes() + 4);
        if (dimension != 128 || file.size != 8ULL + 4ULL * count * dimension)
            throw std::runtime_error("Wrong fbin shape: " + path);
        vectors = reinterpret_cast<const float *>(file.bytes() + 8);
    }
    const float *row(uint32_t id) const { return vectors + static_cast<size_t>(id) * dimension; }
};

struct Graph {
    MappedFile file;
    uint32_t start = 0, count = 0, max_degree = 0;
    const uint32_t *words = nullptr;
    std::vector<size_t> offsets;
    Graph(const std::string &path) : file(path) {
        if (file.size < 24 || file.size % 4 || u64(file.bytes()) != file.size)
            throw std::runtime_error("Invalid official graph header");
        max_degree = u32(file.bytes() + 8);
        start = u32(file.bytes() + 12);
        if (u64(file.bytes() + 16) != 0)
            throw std::runtime_error("Unexpected start ID or frozen points");
        words = reinterpret_cast<const uint32_t *>(file.data);
        size_t cursor = 6;
        while (cursor < file.size / 4) {
            uint32_t degree = words[cursor];
            if (degree > max_degree || cursor + 1ULL + degree > file.size / 4)
                throw std::runtime_error("Invalid graph degree");
            if (offsets.size() == UINT32_MAX) throw std::runtime_error("Too many graph nodes");
            offsets.push_back(cursor + 1);
            cursor += 1ULL + degree;
        }
        count = static_cast<uint32_t>(offsets.size());
        if (cursor != file.size / 4 || count < 10 || start >= count) throw std::runtime_error("Invalid graph count");
    }
    const uint32_t *neighbors(uint32_t id) const { return words + offsets[id]; }
    uint32_t degree(uint32_t id) const { return words[offsets[id] - 1]; }
};

// The float database is opened only for the unquantized control.
struct Scorer {
    std::unique_ptr<FloatBin> base;
    std::unique_ptr<ann_quantization::SearchScorer> codec;
    const float *query = nullptr;
    Scorer(const std::string &kind, const std::string &base_path,
           const std::string &codes, const std::string &model, uint32_t size, uint32_t nodes) {
        if (kind == "exact") {
            base = std::make_unique<FloatBin>(base_path);
            if (base->count != nodes) throw std::runtime_error("Base count does not match graph");
        } else {
            codec = std::make_unique<ann_quantization::SearchScorer>(kind, codes, model, nodes, size);
        }
    }
    void prepare(const float *q) {
        query = q;
        if (codec) codec->prepare(q);
    }
    float score(uint32_t id) const {
        if (codec) return codec->score(id);
        const float *vector = base->row(id);
        float distance = 0;
        for (int d = 0; d < 128; ++d) {
            float diff = vector[d] - query[d];
            distance += diff * diff;
        }
        return distance;
    }
};

struct Searcher {
    const Graph &graph;
    Scorer &scorer;
    std::vector<uint32_t> visited;
    uint32_t epoch = 0;
    diskann::NeighborPriorityQueue pool;
    Searcher(const Graph &g, Scorer &s, uint32_t l)
        : graph(g), scorer(s), visited(g.count, 0), pool(l) {}

    std::vector<uint32_t> search(const float *query, uint32_t l, uint32_t k) {
        scorer.prepare(query);
        if (++epoch == 0) {
            std::fill(visited.begin(), visited.end(), 0);
            epoch = 1;
        }
        pool.clear();
        visited[graph.start] = epoch;
        pool.insert(diskann::Neighbor(graph.start, scorer.score(graph.start)));
        while (pool.has_unexpanded_node()) {
            uint32_t node = pool.closest_unexpanded().id;
            const uint32_t *next = graph.neighbors(node);
            uint32_t degree = graph.degree(node);
            for (uint32_t j = 0; j < degree; ++j) {
                uint32_t id = next[j];
                if (id >= graph.count) throw std::runtime_error("Out-of-range graph edge");
                if (visited[id] != epoch) {
                    visited[id] = epoch;
                    pool.insert(diskann::Neighbor(id, scorer.score(id)));
                }
            }
        }
        if (pool.size() < k) throw std::runtime_error("Too few candidates");
        std::vector<uint32_t> ids;
        for (uint32_t j = 0; j < k; ++j) ids.push_back(pool[j].id);
        return ids;
    }
};

std::map<std::string, std::string> parse_args(int argc, char **argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc || std::strncmp(argv[i], "--", 2))
            throw std::runtime_error("Expected --key value arguments");
        args[argv[i] + 2] = argv[i + 1];
    }
    return args;
}

std::string arg(const std::map<std::string, std::string> &args, const std::string &name) {
    auto found = args.find(name);
    if (found == args.end()) throw std::runtime_error("Missing --" + name);
    return found->second;
}

uint32_t number(const std::string &text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Expected an unsigned integer");
    const auto value = std::stoull(text);
    if (value > UINT32_MAX) throw std::runtime_error("Number outside uint32");
    return static_cast<uint32_t>(value);
}

void write_ids(const std::string &path, const std::vector<uint32_t> &ids,
               uint32_t count, uint32_t k) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write " + path);
    out.write(reinterpret_cast<const char *>(&count), 4);
    out.write(reinterpret_cast<const char *>(&k), 4);
    out.write(reinterpret_cast<const char *>(ids.data()), ids.size() * 4);
}

} // namespace

int main(int argc, char **argv) {
    try {
        auto args = parse_args(argc, argv);
        FloatBin queries(arg(args, "query"));
        if (queries.count < 1)
            throw std::runtime_error("Expected nonempty query fbin");
        Graph graph(arg(args, "graph"));
        std::string kind_name = arg(args, "kind");
        uint32_t l = number(arg(args, "L"));
        uint32_t k = number(arg(args, "K"));
        uint32_t count = number(arg(args, "limit"));
        uint32_t warmup = number(arg(args, "warmup"));
        uint32_t repeats = number(arg(args, "repeats"));
        if (arg(args, "rerank") != "0") throw std::runtime_error("Experiment A forbids rerank");
        if (!k || k > l || l > graph.count || !count || count > queries.count || warmup > count || !repeats)
            throw std::runtime_error("Invalid K, L, limit, or repeats");
        Scorer scorer(kind_name, arg(args, "base"), arg(args, "codes"), arg(args, "model"),
                      number(arg(args, "size")), graph.count);
        Searcher searcher(graph, scorer, l);
        for (uint32_t i = 0; i < warmup; ++i)
            searcher.search(queries.row(i % count), l, k);

        std::vector<double> qpm;
        std::vector<uint32_t> labels;
        labels.reserve(static_cast<size_t>(count) * k);
        for (uint32_t run = 0; run < repeats; ++run) {
            if (run == 0) labels.clear();
            auto start = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < count; ++i) {
                auto ids = searcher.search(queries.row(i), l, k);
                if (run == 0) labels.insert(labels.end(), ids.begin(), ids.end());
            }
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            qpm.push_back(60.0 * count / seconds);
        }
        write_ids(arg(args, "output"), labels, count, k);
        auto ordered = qpm;
        std::sort(ordered.begin(), ordered.end());
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "{\"kind\":\"" << kind_name << "\",\"L\":" << l
                  << ",\"K\":" << k << ",\"queries\":" << count
                  << ",\"warmup\":" << warmup << ",\"repeats\":" << repeats
                  << ",\"rerank\":false,\"database_float_bytes_loaded\":" << (kind_name == "exact" ? static_cast<uint64_t>(graph.count) * 128 * 4 : 0)
                  << ",\"code_size\":" << arg(args, "size")
                  << ",\"median_qpm\":" << ordered[ordered.size() / 2] << ",\"qpm\":[";
        for (size_t i = 0; i < qpm.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << qpm[i];
        }
        std::cout << "]}" << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}

