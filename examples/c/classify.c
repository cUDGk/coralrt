/* coralrt from C: classify one raw input (the model's input tensor as bytes) and print the top class.
 *
 *   clang -O2 -I ../../include classify.c ../../target/release/coralrt.dll.lib -o classify.exe   (Windows)
 *   cc -O2 -I ../../include classify.c -L ../../target/release -lcoralrt -o classify             (Linux / macOS)
 *   classify model_edgetpu.tflite input.bin [labels.txt]
 */
#define _CRT_SECURE_NO_WARNINGS /* plain fopen, portable C */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "coralrt.h"

static unsigned char* read_file(const char* path, size_t* len) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  *len = (size_t)ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char* buf = malloc(*len);
  if (buf && fread(buf, 1, *len, f) != *len) {
    free(buf);
    buf = NULL;
  }
  fclose(f);
  return buf;
}

/* the label on line `index` of a one-label-per-line file */
static void print_label(const char* path, int index) {
  char line[512];
  FILE* f = path ? fopen(path, "r") : NULL;
  for (int i = 0; f && fgets(line, sizeof line, f); ++i) {
    if (i == index) {
      line[strcspn(line, "\r\n")] = 0;
      printf(" (%s)", line);
      break;
    }
  }
  if (f) fclose(f);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s model_edgetpu.tflite input.bin [labels.txt]\n", argv[0]);
    return 2;
  }
  CoralInterpreter* it = coral_interpreter_new(argv[1], NULL);
  if (!it) {
    fprintf(stderr, "coralrt: %s\n", coral_last_error());
    return 1;
  }
  size_t n;
  unsigned char* input = read_file(argv[2], &n);
  if (!input || coral_set_input(it, 0, input, n) != 0) {
    fprintf(stderr, "input: %s\n", input ? coral_last_error() : "cannot read the file");
    return 1;
  }
  int32_t out = coral_output_tensor(it, 0);
  ptrdiff_t classes = coral_tensor_bytes(it, out);
  if (coral_tensor_type(it, out) != CORAL_UINT8) {
    fprintf(stderr, "expected a uint8 classifier output\n");
    return 1;
  }
  clock_t start = clock();
  const int runs = 100;
  for (int r = 0; r < runs; ++r) {
    if (coral_interpreter_invoke(it) != 0) {
      fprintf(stderr, "invoke: %s\n", coral_last_error());
      return 1;
    }
  }
  double ms = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC / runs;
  const uint8_t* scores = coral_tensor_data(it, out);
  int best = 0;
  for (int i = 1; i < classes; ++i)
    if (scores[i] > scores[best]) best = i;
  float scale = coral_tensor_quant_scale(it, out, 0);
  int64_t zero = coral_tensor_quant_zero_point(it, out, 0);
  printf("class %d score %.5f", best, scale * (float)(scores[best] - zero));
  print_label(argc > 3 ? argv[3] : NULL, best);
  printf(", %.2f ms per inference (coralrt %s)\n", ms, coral_version());
  free(input);
  coral_interpreter_free(it);
  return 0;
}
