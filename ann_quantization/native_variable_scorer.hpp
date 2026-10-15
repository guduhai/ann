#pragma once

// Same ADC/PCA formulas as native_scorer.hpp, with an explicit code size.
// Reuses the existing read-only mmap utilities; no original vectors are loaded.
#include "native_scorer.hpp"
#include <vector>

namespace ann_quantization {
class VariableScorer {
 public:
    VariableScorer(const std::string &kind, const std::string &prepared,
                   const std::string &model_dir, uint32_t count, uint32_t size)
        : kind_(kind), count_(count), size_(size) {
        if (kind == "pq") {
            if (!size || size > 128 || 128 % size)
                throw std::runtime_error("PQ size must divide 128");
            width_ = 128 / size;
            stride_ = size;
            codebooks_ = std::make_unique<MappedFile>(model_dir + "/codebooks.bin");
            expect_size(codebooks_, 128 * 256 * 4);
            lookup_.resize(size * 256);
        } else if (kind == "pca") {
            if (!size || size > 128) throw std::runtime_error("PCA size must be 1..128");
            stride_ = size * 4;
            mean_ = std::make_unique<MappedFile>(model_dir + "/mean.bin");
            basis_ = std::make_unique<MappedFile>(model_dir + "/basis.bin");
            expect_size(mean_, 128 * 4);
            expect_size(basis_, 128 * size * 4);
            projected_.resize(size);
        } else {
            throw std::runtime_error("VariableScorer supports PQ/PCA only");
        }
        codes_ = std::make_unique<MappedFile>(prepared + "/codes.npy");
        payload_ = npy_data(*codes_, static_cast<size_t>(count) * stride_);
    }

    void prepare(const float *query) {
        ready_ = true;
        if (kind_ == "pca") {
            const float *mu = reinterpret_cast<const float *>(mean_->data());
            const float *projection = reinterpret_cast<const float *>(basis_->data());
            for (uint32_t j = 0; j < size_; ++j) {
                float value = 0;
                for (uint32_t d = 0; d < 128; ++d)
                    value += (query[d] - mu[d]) * projection[d * size_ + j];
                projected_[j] = value;
            }
        } else {
            const float *books = reinterpret_cast<const float *>(codebooks_->data());
            for (uint32_t sub = 0; sub < size_; ++sub) {
                for (uint32_t center = 0; center < 256; ++center) {
                    float distance = 0;
                    for (uint32_t d = 0; d < width_; ++d) {
                        const float diff = books[(sub * 256 + center) * width_ + d] - query[sub * width_ + d];
                        distance += diff * diff;
                    }
                    lookup_[sub * 256 + center] = distance;
                }
            }
        }
    }

    float score(uint32_t id) const {
        if (id >= count_ || !ready_) throw std::runtime_error("Invalid score request");
        const uint8_t *code = payload_ + static_cast<size_t>(id) * stride_;
        float distance = 0;
        if (kind_ == "pca") {
            const float *values = reinterpret_cast<const float *>(code);
            for (uint32_t j = 0; j < size_; ++j) {
                float diff = values[j] - projected_[j];
                distance += diff * diff;
            }
        } else {
            for (uint32_t sub = 0; sub < size_; ++sub)
                distance += lookup_[sub * 256 + code[sub]];
        }
        return distance;
    }

 private:
    static void expect_size(const std::unique_ptr<MappedFile> &file, size_t expected) {
        if (file->size() != expected) throw std::runtime_error("Wrong variable codec model size");
    }
    std::string kind_;
    uint32_t count_, size_, width_ = 0;
    size_t stride_ = 0;
    bool ready_ = false;
    std::unique_ptr<MappedFile> codes_, mean_, basis_, codebooks_;
    const uint8_t *payload_ = nullptr;
    std::vector<float> projected_, lookup_;
};
}  // namespace ann_quantization
