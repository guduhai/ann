#pragma once
// Distance kernels copied from this project's C++ implementation.
// Full-precision base vectors are absent from compressed modes.
#include <algorithm>
#include <cstdint>
#include <vector>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
struct Config {
    int32_t n{}, d{}, mode{}, width{}, centroids{}, clusters{};
    const void* codes{};
    const float *scale{}, *mean{}, *basis{}, *codebooks{}, *centers{};
};
static float l2(const float* x, const float* y, int d) {
    float sum = 0;
    int j = 0;
#if defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0);
    for (; j + 4 <= d; j += 4) {
        auto diff = vsubq_f32(vld1q_f32(x+j), vld1q_f32(y+j));
        acc = vaddq_f32(acc, vmulq_f32(diff, diff));
    }
    sum = vaddvq_f32(acc);
#endif
    for (; j < d; ++j) { float diff = x[j] - y[j]; sum += diff*diff; }
    return sum;
}

static float residual_l2(const int8_t* code, const float* scale,
                         const float* center, const float* q, int d) {
    float sum = 0;
    int j = 0;
#if defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0);
    for (; j + 8 <= d; j += 8) {
        int16x8_t i16 = vmovl_s8(vld1_s8(code+j));
        float32x4_t lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16)));
        float32x4_t hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(i16)));
        lo = vmulq_f32(lo, vld1q_f32(scale+j));
        hi = vmulq_f32(hi, vld1q_f32(scale+j+4));
        if (center) {
            lo = vaddq_f32(lo, vld1q_f32(center+j));
            hi = vaddq_f32(hi, vld1q_f32(center+j+4));
        }
        lo = vsubq_f32(lo, vld1q_f32(q+j));
        hi = vsubq_f32(hi, vld1q_f32(q+j+4));
        acc = vaddq_f32(acc, vmulq_f32(lo, lo));
        acc = vaddq_f32(acc, vmulq_f32(hi, hi));
    }
    sum = vaddvq_f32(acc);
#endif
    for (; j < d; ++j) {
        float diff = float(code[j])*scale[j];
        if (center) diff += center[j];
        diff -= q[j]; sum += diff*diff;
    }
    return sum;
}

template<int Mode> struct Distance {
    const Config& c;
    const float* q;
    std::vector<float> state;
    explicit Distance(const Config& config) : c(config), q(nullptr) {
        if constexpr (Mode == 2) state.resize(c.width);
        if constexpr (Mode == 3) state.resize(c.width*c.centroids);
    }
    void prepare(const float* query) {
        q = query;
        if constexpr (Mode == 2) {
            std::fill(state.begin(), state.end(), 0);
            // Row-major PCA basis; SIMD runs over output components.
            for (int j = 0; j < c.d; ++j) {
                float value = q[j]-c.mean[j]; int t = 0;
#if defined(__ARM_NEON)
                for (; t+4 <= c.width; t+=4) {
                    auto add = vmulq_n_f32(vld1q_f32(c.basis+j*c.width+t), value);
                    vst1q_f32(state.data()+t, vaddq_f32(vld1q_f32(state.data()+t), add));
                }
#endif
                for (; t < c.width; ++t) state[t] += value*c.basis[j*c.width+t];
            }
        }
        if constexpr (Mode == 3) {
            int subdim = c.d/c.width;
            for (int s = 0; s < c.width; ++s)
                for (int k = 0; k < c.centroids; ++k)
                    state[s*c.centroids+k] = l2(c.codebooks+(s*c.centroids+k)*subdim,
                                                q+s*subdim, subdim);
        }
    }
    float operator()(int id) const {
        if constexpr (Mode == 0) return l2(static_cast<const float*>(c.codes)+int64_t(id)*c.d, q, c.d);
        if constexpr (Mode == 1)
            return residual_l2(static_cast<const int8_t*>(c.codes)+int64_t(id)*c.d,
                               c.scale, nullptr, q, c.d);
        if constexpr (Mode == 2)
            return l2(static_cast<const float*>(c.codes)+int64_t(id)*c.width,
                       state.data(), c.width);
        if constexpr (Mode == 3) {
            auto p = static_cast<const uint8_t*>(c.codes)+int64_t(id)*c.width;
            float sum = 0;
            for (int s = 0; s < c.width; ++s) sum += state[s*c.centroids+p[s]];
            return sum;
        }
        if constexpr (Mode == 4) {
            auto p = static_cast<const uint8_t*>(c.codes)+int64_t(id)*(c.d+1);
            int group = p[0];
            return residual_l2(reinterpret_cast<const int8_t*>(p+1),
                               c.scale+group*c.d, c.centers+group*c.d, q, c.d);
        }
        return 0;
    }
};

