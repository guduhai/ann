#include "util.h"

#include <malloc.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <new>
#include <stdexcept>

namespace efanna2e {

void GenRandom(std::mt19937& rng, unsigned* addr, unsigned size, unsigned N) {
  if (!N || size >= N) throw std::invalid_argument("GenRandom requires size < N and N > 0");
  for (unsigned i = 0; i < size; ++i) {
    addr[i] = rng() % (N - size);
  }
  std::sort(addr, addr + size);
  for (unsigned i = 1; i < size; ++i) {
    if (addr[i] <= addr[i - 1]) {
      addr[i] = addr[i - 1] + 1;
    }
  }
  unsigned off = rng() % N;
  for (unsigned i = 0; i < size; ++i) {
    addr[i] = (addr[i] + off) % N;
  }
}

float* load_data(const char* filename, unsigned& num, unsigned& dim) {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "Open file error" << std::endl;
    exit(-1);
  }

  in.read((char*)&dim, 4);

  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);

  float* data = new float[(size_t)num * (size_t)dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * sizeof(float));
  }
  in.close();

  return data;
}

float* data_align(float* data_ori, unsigned point_num, unsigned& dim) {
#ifdef __GNUC__
#ifdef __AVX__
#define DATA_ALIGN_FACTOR 8
#else
#ifdef __SSE2__
#define DATA_ALIGN_FACTOR 4
#else
#define DATA_ALIGN_FACTOR 1
#endif
#endif
#endif
  float* data_new = 0;
  unsigned new_dim =
      (dim + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;
#ifdef __APPLE__
  data_new = new float[(size_t)new_dim * (size_t)point_num];
#else
  data_new =
      (float*)memalign(std::max<size_t>(DATA_ALIGN_FACTOR * 4, sizeof(void*)),
                       (size_t)point_num * (size_t)new_dim * sizeof(float));
#endif

  if (!data_new) throw std::bad_alloc();
  for (size_t i = 0; i < point_num; i++) {
    memcpy(data_new + i * new_dim, data_ori + i * dim, dim * sizeof(float));
    memset(data_new + i * new_dim + dim, 0, (new_dim - dim) * sizeof(float));
  }

  dim = new_dim;

  // load_data allocates the source with new[] on every platform.
  delete[] data_ori;

  return data_new;
}

}  // namespace efanna2e
