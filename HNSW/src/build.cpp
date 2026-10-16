// Native CLI for official hnswlib 0.8.0. Graph construction is not reimplemented.
#include "hnswlib/hnswlib.h"
#include "parallel_for.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static size_t number(const std::string &text, bool zero_allowed = false) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("Expected an unsigned integer: " + text);
    size_t used = 0;
    auto n = std::stoull(text, &used);
    if (used != text.size() || n > std::numeric_limits<size_t>::max() || (!zero_allowed && n == 0))
        throw std::invalid_argument("Integer out of range: " + text);
    return static_cast<size_t>(n);
}

static double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> options = {
            {"--m", "16"}, {"--ef-construction", "200"}, {"--seed", "100"},
            {"--build-threads", "4"}, {"--dataset", ""}, {"--index", ""}, {"--result", ""}};
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--help") {
                std::cout << "ann_build --dataset BASE.fvecs --index OUTPUT.bin "
                             "[--m 16 --ef-construction 200 --seed 100 --build-threads 4] "
                             "[--result REPORT.json]\n";
                return 0;
            }
            if (!options.count(key) || i + 1 == argc)
                throw std::invalid_argument("Unknown or incomplete option: " + key);
            options[key] = argv[++i];
        }
        if (options["--dataset"].empty() || options["--index"].empty())
            throw std::invalid_argument("--dataset and --index are required; input format is fvecs.");
        const size_t m = number(options["--m"]);
        const size_t ef = number(options["--ef-construction"]);
        const size_t seed = number(options["--seed"], true);
        const size_t requested_threads = number(options["--build-threads"]);
        if (m > 10000 || requested_threads > 1024)
            throw std::invalid_argument("M must be <= 10000 and threads <= 1024.");
        const fs::path output = options["--index"];
        const fs::path report = options["--result"];
        if (fs::exists(output) || (!report.empty() && fs::exists(report)))
            throw std::runtime_error("Refusing to overwrite an existing index or report.");
        if (!report.empty() && fs::absolute(output).lexically_normal() == fs::absolute(report).lexically_normal())
            throw std::runtime_error("Index and report paths must be different.");
        const uint16_t endian = 1;
        if (*reinterpret_cast<const uint8_t *>(&endian) != 1)
            throw std::runtime_error("This fvecs/index workflow requires a little-endian host.");

        const auto begin = Clock::now();
        std::ifstream input(options["--dataset"], std::ios::binary);
        if (!input) throw std::runtime_error("Cannot open input fvecs file.");
        int32_t dim = 0;
        input.read(reinterpret_cast<char *>(&dim), sizeof(dim));
        if (!input || dim <= 0 || dim > 65536)
            throw std::runtime_error("Invalid fvecs dimension.");
        const uintmax_t record_bytes = (uintmax_t(dim) + 1) * sizeof(float);
        const uintmax_t file_bytes = fs::file_size(options["--dataset"]);
        if (file_bytes == 0 || file_bytes % record_bytes != 0)
            throw std::runtime_error("Truncated fvecs input.");
        const uintmax_t rows = file_bytes / record_bytes;
        if (rows > std::numeric_limits<uint32_t>::max() ||
            rows > std::numeric_limits<size_t>::max() / size_t(dim) / sizeof(float))
            throw std::runtime_error("Dataset exceeds supported size.");
        const size_t count = static_cast<size_t>(rows);
        std::vector<float> vectors(count * size_t(dim));
        input.seekg(0);
        for (size_t row = 0; row < count; ++row) {
            int32_t row_dim = 0;
            input.read(reinterpret_cast<char *>(&row_dim), sizeof(row_dim));
            input.read(reinterpret_cast<char *>(vectors.data() + row * dim), dim * sizeof(float));
            if (!input || row_dim != dim) throw std::runtime_error("Invalid fvecs record.");
        }
        const auto loaded = Clock::now();
        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float> index(&space, count, m, ef, seed, false);
        const size_t threads = count <= requested_threads * 4 ? 1 : requested_threads;
        const auto insert_begin = Clock::now();
        // Match bindings.cpp:addItems: insert the entry point first, then ParallelFor.
        index.addPoint(vectors.data(), 0, false);
        ParallelFor(1, count, threads, [&](size_t row, size_t) {
            index.addPoint(vectors.data() + row * dim, row, false);
        });
        const auto inserted = Clock::now();
        if (index.getCurrentElementCount() != count) throw std::runtime_error("Incomplete index.");
        if (output.has_parent_path()) fs::create_directories(output.parent_path());
        index.saveIndex(output.string());
        if (!fs::is_regular_file(output) || fs::file_size(output) != index.indexFileSize())
            throw std::runtime_error("Index save failed.");
        const auto saved = Clock::now();
        const double insert_seconds = seconds(insert_begin, inserted);
        const double total_seconds = seconds(begin, saved);
        std::ostringstream json;
        json << std::setprecision(12)
             << "{\"phase\":\"build\",\"implementation\":\"hnswlib-0.8.0-native-cli\","
             << "\"input_format\":\"fvecs\",\"vectors\":" << count
             << ",\"dimensions\":" << dim << ",\"m\":" << m
             << ",\"ef_construction\":" << ef << ",\"seed\":" << seed
             << ",\"build_threads\":" << requested_threads << ",\"effective_threads\":" << threads
             << ",\"load_seconds\":" << seconds(begin, loaded)
             << ",\"insert_seconds\":" << insert_seconds
             << ",\"save_seconds\":" << seconds(inserted, saved)
             << ",\"total_seconds\":" << total_seconds
             << ",\"insert_vectors_per_second\":" << count / insert_seconds
             << ",\"total_vectors_per_second\":" << count / total_seconds
             << ",\"index_bytes\":" << fs::file_size(output) << "}";
        if (!report.empty()) {
            if (report.has_parent_path()) fs::create_directories(report.parent_path());
            std::ofstream stream(report);
            stream << json.str() << '\n';
            stream.close();
            if (!stream) throw std::runtime_error("Report write failed.");
        }
        std::cout << json.str() << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ann_build: " << error.what() << '\n';
        return 1;
    }
}
