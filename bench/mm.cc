#include "benchmark/benchmark.h"
#include "deepgreen/matrix_mm.hh"
#include "deepgreen/matrix_mm.cuh"

static void BM_NaiveMatrixMult(benchmark::State& state) {
  while (state.KeepRunning()) {
    state.PauseTiming();
    auto x = deepgreen::matrix<float>(state.range(0), state.range(0), 1.5f);
    auto y = deepgreen::matrix<float>(state.range(0), state.range(0), 1.5f);
    state.ResumeTiming();
    auto z = deepgreen::naive_mm(x, y);
  }
}
BENCHMARK(BM_NaiveMatrixMult)
    ->Args({100})
    ->Args({300})
    ->Args({500})
    ->Args({700})
    ->Args({900})
    ->Args({1000});

static void BM_CudaMatrixMult(benchmark::State& state) {
  while (state.KeepRunning()) {
    state.PauseTiming();
    auto x = deepgreen::matrix<float>(state.range(0), state.range(0), 1.5f);
    auto y = deepgreen::matrix<float>(state.range(0), state.range(0), 1.5f);
    state.ResumeTiming();
    auto z = deepgreen::cuda_mm(x, y);
  }
}
BENCHMARK(BM_CudaMatrixMult)
    ->Args({100})
    ->Args({300})
    ->Args({500})
    ->Args({700})
    ->Args({900})
    ->Args({1000});

BENCHMARK_MAIN();
