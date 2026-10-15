// CPU search on the frozen official graph; no full-precision reranking.
// Distances are supplied by ann_quantization; graph traversal is unchanged.
#include <hnswlib/hnswlib.h>
#include "native_search_scorer.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

using Index = hnswlib::HierarchicalNSW<float>;
struct Context {
    ann_quantization::SearchScorer *scorer = nullptr;
    const Index *index = nullptr;
    bool compact = false;
    uint64_t calls = 0;
};

float score_code(const void *, const void *stored, const void *opaque) {
    auto &context = *static_cast<Context *>(const_cast<void *>(opaque));
    uint32_t id;
    if (context.compact) {
        std::memcpy(&id, stored, sizeof(id));
    } else {
        const char *first = context.index->data_level0_memory_ + context.index->offsetData_;
        ptrdiff_t offset = static_cast<const char *>(stored) - first;
        if (offset < 0 || offset % context.index->size_data_per_element_)
            throw std::runtime_error("Invalid original HNSW vector pointer");
        id = static_cast<uint32_t>(offset / context.index->size_data_per_element_);
    }
    ++context.calls;
    return context.scorer->score(id);
}

class CodeIdSpace : public hnswlib::SpaceInterface<float> {
    Context &context_;
 public:
    explicit CodeIdSpace(Context &context) : context_(context) {}
    size_t get_data_size() override { return sizeof(uint32_t); }
    hnswlib::DISTFUNC<float> get_dist_func() override { return score_code; }
    void *get_dist_func_param() override { return &context_; }
};

std::vector<float> load_queries(const std::string &path, uint32_t &rows) {
    std::ifstream source(path, std::ios::binary | std::ios::ate);
    if (!source || source.tellg() != 10000 * 129 * 4)
        throw std::runtime_error("Expected all 10000 SIFT query fvecs");
    rows = 10000;
    source.seekg(0);
    std::vector<float> queries(static_cast<size_t>(rows) * 128);
    for (uint32_t i = 0; i < rows; ++i) {
        uint32_t dimension;
        source.read(reinterpret_cast<char *>(&dimension), 4);
        if (dimension != 128) throw std::runtime_error("Invalid query dimension");
        source.read(reinterpret_cast<char *>(queries.data() + static_cast<size_t>(i) * 128), 128 * 4);
    }
    if (!source) throw std::runtime_error("Short query file");
    return queries;
}

uint32_t number(const char *value) {
    const unsigned long result = std::stoul(value);
    if (result > UINT32_MAX) throw std::runtime_error("Number outside uint32");
    return static_cast<uint32_t>(result);
}

int main(int argc, char **argv) {
    try {
        if (argc != 13)
            throw std::runtime_error("Usage: quant_only_search MODE INDEX QUERY KIND CODES MODEL W COUNT WARMUP REPEATS OUTPUT CODE_SIZE");
        const std::string mode = argv[1], kind = argv[4];
        const bool compact = mode == "compact", quantized = mode != "float";
        if (mode != "compact" && mode != "reference" && mode != "float")
            throw std::runtime_error("Mode must be compact/reference/float");
        if ((!quantized && kind != "exact") || (quantized && kind == "exact"))
            throw std::runtime_error("Mode/kind mismatch");
        const uint32_t width = number(argv[7]), count = number(argv[8]);
        const uint32_t warmup = number(argv[9]), repeats = number(argv[10]);
        if (width < 10 || width > 1000000 || !count || count > 10000 || warmup > count || !repeats)
            throw std::runtime_error("Invalid search parameters");
        uint32_t available;
        const auto queries = load_queries(argv[3], available);
        std::unique_ptr<ann_quantization::SearchScorer> scorer;
        if (quantized)
            scorer = std::make_unique<ann_quantization::SearchScorer>(kind, argv[5], argv[6], 1000000, number(argv[12]));
        Context context{scorer.get(), nullptr, compact, 0};
        hnswlib::L2Space float_space(128);
        CodeIdSpace id_space(context);
        Index index(compact ? static_cast<hnswlib::SpaceInterface<float> *>(&id_space) : &float_space, argv[2]);
        context.index = &index;
        if (index.getCurrentElementCount() != 1000000 || index.num_deleted_ != 0)
            throw std::runtime_error("Expected one million undeleted graph nodes");
        const size_t payload_bytes = index.label_offset_ - index.offsetData_;
        if (payload_bytes != (compact ? 4 : 128 * 4))
            throw std::runtime_error("Wrong original/compact index payload layout");
        if (compact) {
            for (uint32_t i = 0; i < 1000000; ++i) {
                uint32_t stored;
                std::memcpy(&stored, index.getDataByInternalId(i), sizeof(stored));
                if (stored != i) throw std::runtime_error("Internal code ID mismatch");
            }
        } else if (quantized) {
            index.fstdistfunc_ = score_code;
            index.dist_func_param_ = &context;
        }
        index.setEf(width);
        auto search = [&](uint32_t row) {
            const float *query = queries.data() + static_cast<size_t>(row) * 128;
            if (scorer) scorer->prepare(query);
            auto nearest = index.searchKnn(query, 10);
            if (nearest.size() != 10) throw std::runtime_error("Missing Top-10 results");
            std::vector<uint32_t> ids(10);
            for (int i = 9; i >= 0; --i) {
                ids[i] = static_cast<uint32_t>(nearest.top().second);
                nearest.pop();
            }
            return ids;
        };
        for (uint32_t i = 0; i < warmup; ++i) search(i);
        std::vector<uint32_t> first;
        first.reserve(static_cast<size_t>(count) * 10);
        std::vector<double> qpm;
        context.calls = 0;
        for (uint32_t run = 0; run < repeats; ++run) {
            const auto start = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < count; ++i) {
                const auto ids = search(i);
                if (!run) first.insert(first.end(), ids.begin(), ids.end());
            }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            qpm.push_back(60.0 * count / seconds);
        }
        std::ofstream output(argv[11], std::ios::binary);
        const uint32_t k = 10;
        output.write(reinterpret_cast<const char *>(&count), 4);
        output.write(reinterpret_cast<const char *>(&k), 4);
        output.write(reinterpret_cast<const char *>(first.data()), first.size() * 4);
        if (!output) throw std::runtime_error("Cannot write results");
        struct rusage resources{};
        getrusage(RUSAGE_SELF, &resources);
        auto ordered = qpm;
        std::sort(ordered.begin(), ordered.end());
        std::cout << std::fixed << std::setprecision(6)
                  << "{\"mode\":\"" << mode << "\",\"kind\":\"" << kind
                  << "\",\"W\":" << width << ",\"queries\":" << count
                  << ",\"code_size\":" << number(argv[12]) << ",\"warmup\":" << warmup << ",\"repeats\":" << repeats
                  << ",\"exact_rerank\":false,\"database_float_bytes_loaded\":" << (compact ? 0 : 512000000)
                  << ",\"index_record_bytes\":" << index.size_data_per_element_
                  << ",\"peak_rss_kib\":" << resources.ru_maxrss
                  << ",\"score_calls_per_query\":" << double(context.calls) / (count * repeats)
                  << ",\"median_qpm\":" << ordered[ordered.size() / 2] << ",\"qpm\":[";
        for (size_t i = 0; i < qpm.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << qpm[i];
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
