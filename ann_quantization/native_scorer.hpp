#pragma once

// Search-only scoring for the four already fitted Python codecs.
// Native graph traversals can call score(vertex_id); graph construction is untouched.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace ann_quantization {

class MappedFile {
 public:
    explicit MappedFile(const std::string &path) {
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("Cannot open " + path);
        struct stat info {};
        if (fstat(fd_, &info) || info.st_size <= 0) {
            close(fd_);
            throw std::runtime_error("Cannot stat " + path);
        }
        size_ = static_cast<size_t>(info.st_size);
        void *pointer = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (pointer == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("Cannot mmap " + path);
        }
        data_ = static_cast<const uint8_t *>(pointer);
    }
    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;
    ~MappedFile() {
        if (data_) munmap(const_cast<uint8_t *>(data_), size_);
        if (fd_ >= 0) close(fd_);
    }
    const uint8_t *data() const { return data_; }
    size_t size() const { return size_; }

 private:
    int fd_ = -1;
    size_t size_ = 0;
    const uint8_t *data_ = nullptr;
};

inline const uint8_t *npy_data(const MappedFile &file, size_t expected_bytes) {
    const uint8_t *p = file.data();
    if (file.size() < 12 || std::memcmp(p, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("Invalid .npy header");
    size_t offset;
    if (p[6] == 1) {
        offset = 10 + static_cast<size_t>(p[8]) + (static_cast<size_t>(p[9]) << 8);
    } else if (p[6] == 2 || p[6] == 3) {
        offset = 12 + static_cast<size_t>(p[8]) + (static_cast<size_t>(p[9]) << 8) +
                 (static_cast<size_t>(p[10]) << 16) + (static_cast<size_t>(p[11]) << 24);
    } else {
        throw std::runtime_error("Unsupported .npy version");
    }
    if (offset > file.size() || file.size() - offset != expected_bytes)
        throw std::runtime_error("Wrong .npy payload size");
    return p + offset;
}

class Scorer {
 public:
    Scorer(const std::string &kind, const std::string &prepared,
           const std::string &model_dir, uint32_t count)
        : kind_(kind), count_(count) {
        if (kind == "int8") stride_ = 128;
        else if (kind == "pca") stride_ = 64 * 4;
        else if (kind == "pq") stride_ = 16;
        else if (kind == "local") stride_ = 129;
        else throw std::runtime_error("Unknown codec");
        codes_ = std::make_unique<MappedFile>(prepared + "/codes.npy");
        payload_ = npy_data(*codes_, static_cast<size_t>(count) * stride_);
        if (kind == "int8") {
            scale_ = std::make_unique<MappedFile>(model_dir + "/scale.bin");
            expect_size(scale_, 128 * 4);
        } else if (kind == "pca") {
            mean_ = std::make_unique<MappedFile>(model_dir + "/mean.bin");
            basis_ = std::make_unique<MappedFile>(model_dir + "/basis.bin");
            expect_size(mean_, 128 * 4);
            expect_size(basis_, 128 * 64 * 4);
        } else if (kind == "pq") {
            codebooks_ = std::make_unique<MappedFile>(model_dir + "/codebooks.bin");
            expect_size(codebooks_, 16 * 256 * 8 * 4);
        } else {
            centers_ = std::make_unique<MappedFile>(model_dir + "/centers.bin");
            scale_ = std::make_unique<MappedFile>(model_dir + "/scale.bin");
            expect_size(centers_, 32 * 128 * 4);
            expect_size(scale_, 32 * 128 * 4);
        }
    }

    void prepare(const float *query) {
        query_ = query;
        if (kind_ == "pca") {
            const float *mu = reinterpret_cast<const float *>(mean_->data());
            const float *projection = reinterpret_cast<const float *>(basis_->data());
            for (int j = 0; j < 64; ++j) {
                float value = 0;
                for (int d = 0; d < 128; ++d)
                    value += (query[d] - mu[d]) * projection[d * 64 + j];
                pca_query_[j] = value;
            }
        } else if (kind_ == "pq") {
            const float *books = reinterpret_cast<const float *>(codebooks_->data());
            for (int sub = 0; sub < 16; ++sub) {
                for (int center = 0; center < 256; ++center) {
                    float distance = 0;
                    for (int d = 0; d < 8; ++d) {
                        float diff = books[(sub * 256 + center) * 8 + d] -
                                     query[sub * 8 + d];
                        distance += diff * diff;
                    }
                    pq_lookup_[sub][center] = distance;
                }
            }
        }
    }

    float score(uint32_t id) const {
        if (id >= count_ || !query_) throw std::runtime_error("Invalid score request");
        const uint8_t *code = payload_ + static_cast<size_t>(id) * stride_;
        float distance = 0;
        if (kind_ == "int8") {
            const float *scales = reinterpret_cast<const float *>(scale_->data());
            const int8_t *bytes = reinterpret_cast<const int8_t *>(code);
            for (int d = 0; d < 128; ++d) {
                float diff = static_cast<float>(bytes[d]) * scales[d] - query_[d];
                distance += diff * diff;
            }
        } else if (kind_ == "pca") {
            const float *values = reinterpret_cast<const float *>(code);
            for (int d = 0; d < 64; ++d) {
                float diff = values[d] - pca_query_[d];
                distance += diff * diff;
            }
        } else if (kind_ == "pq") {
            for (int sub = 0; sub < 16; ++sub)
                distance += pq_lookup_[sub][code[sub]];
        } else {
            const uint8_t cluster = code[0];
            if (cluster >= 32) throw std::runtime_error("Invalid local cluster");
            const float *center = reinterpret_cast<const float *>(centers_->data()) +
                                  static_cast<size_t>(cluster) * 128;
            const float *scales = reinterpret_cast<const float *>(scale_->data()) +
                                  static_cast<size_t>(cluster) * 128;
            const int8_t *residual = reinterpret_cast<const int8_t *>(code + 1);
            for (int d = 0; d < 128; ++d) {
                float diff = center[d] + scales[d] * static_cast<float>(residual[d]) -
                             query_[d];
                distance += diff * diff;
            }
        }
        return distance;
    }

 private:
    static void expect_size(const std::unique_ptr<MappedFile> &file, size_t expected) {
        if (file->size() != expected) throw std::runtime_error("Wrong codec model size");
    }
    std::string kind_;
    uint32_t count_;
    size_t stride_ = 0;
    std::unique_ptr<MappedFile> codes_, scale_, mean_, basis_, codebooks_, centers_;
    const uint8_t *payload_ = nullptr;
    const float *query_ = nullptr;
    float pca_query_[64]{};
    float pq_lookup_[16][256]{};
};

} // namespace ann_quantization
