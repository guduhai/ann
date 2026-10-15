#pragma once

// Only the BLAS/LAPACKE calls used by the upstream memory-index build are
// mapped here; graph construction and pruning code are left unchanged.
#include <cblas.h>
#include <lapacke.h>

using MKL_INT = int;
extern "C" void openblas_set_num_threads(int);
inline void mkl_set_num_threads(int n) { openblas_set_num_threads(n); }
