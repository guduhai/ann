#pragma once

// The upstream x86 intrinsic surface is translated by SIMDe on ARM64.
// USE_AVX2 is translated through SIMDe to ARM NEON by the separate CMake build.
#ifndef SIMDE_ENABLE_NATIVE_ALIASES
#define SIMDE_ENABLE_NATIVE_ALIASES
#endif
#include <simde/x86/sse.h>
#include <simde/x86/sse2.h>
#include <simde/x86/ssse3.h>
#include <simde/x86/sse4.1.h>
#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>
