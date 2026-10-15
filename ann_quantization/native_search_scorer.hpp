#pragma once
// Dispatch only: reuse the frozen four codec formulas without changing traversal.
#include "native_variable_scorer.hpp"
namespace ann_quantization {
class SearchScorer {
    std::unique_ptr<Scorer> fixed_;
    std::unique_ptr<VariableScorer> variable_;
 public:
    SearchScorer(const std::string& kind, const std::string& codes,
                 const std::string& model, uint32_t count, uint32_t size) {
        if (kind == "int8" || kind == "local")
            fixed_ = std::make_unique<Scorer>(kind, codes, model, count);
        else variable_ = std::make_unique<VariableScorer>(kind, codes, model, count, size);
    }
    void prepare(const float* q) {
        if (fixed_) fixed_->prepare(q); else variable_->prepare(q);
    }
    float score(uint32_t id) const {
        return fixed_ ? fixed_->score(id) : variable_->score(id);
    }
};
}
