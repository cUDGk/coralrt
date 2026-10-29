// std::partial_sort of indices by decreasing value (as TFLite's DecreasingPartialArgSort), on cases with many ties.
// Prints one line per case: n k then the first k indices. The same LCG drives the Rust side.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

static uint64_t state = 12345;
static uint32_t next() {
  state = state * 6364136223846793005ULL + 1442695040888963407ULL;
  return (uint32_t)(state >> 33);
}

int main() {
  for (int c = 0; c < 2000; ++c) {
    int n = 1 + next() % 300;
    int k = 1 + next() % n;
    int distinct = 1 + next() % 12;
    std::vector<float> v(n);
    for (auto& x : v) x = (float)(next() % distinct) * 0.25f;
    std::vector<int> ix(n);
    std::iota(ix.begin(), ix.end(), 0);
    std::partial_sort(ix.begin(), ix.begin() + k, ix.end(), [&v](int i, int j) { return v[i] > v[j]; });
    printf("%d %d", n, k);
    for (int i = 0; i < k; ++i) printf(" %d", ix[i]);
    printf("\n");
  }
}
