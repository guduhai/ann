#pragma once
#include <cstddef>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
namespace efanna2e {
enum Metric { L2, INNER_PRODUCT, FAST_L2 };
class Distance { public: virtual ~Distance()=default; virtual float compare(const float*,const float*,unsigned) const=0; };
class DistanceL2 : public Distance {
public:
    float compare(const float* a,const float* b,unsigned d) const override {
        float sum=0; unsigned j=0;
#if defined(__ARM_NEON)
        auto acc=vdupq_n_f32(0);
        for(;j+4<=d;j+=4){auto x=vsubq_f32(vld1q_f32(a+j),vld1q_f32(b+j));acc=vaddq_f32(acc,vmulq_f32(x,x));}
        sum=vaddvq_f32(acc);
#endif
        for(;j<d;++j){float x=a[j]-b[j];sum+=x*x;} return sum;
    }
};
}
