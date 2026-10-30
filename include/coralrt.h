/* coralrt: runs TensorFlow Lite models, and Coral Edge TPU models (*_edgetpu.tflite) on the Coral USB Accelerator,
 * without libedgetpu or TensorFlow Lite. Link against coralrt.dll / libcoralrt.so / libcoralrt.dylib (built with
 * `cargo build --release`). The stick is reached through libusb-1.0, loaded at run time (on Windows the
 * libusb-1.0.dll and WinUSB driver that the official Edge TPU runtime installs; CORALRT_LIBUSB overrides
 * the library path).
 *
 *   CoralInterpreter* it = coral_interpreter_new("model_edgetpu.tflite", NULL);
 *   if (!it) { fprintf(stderr, "%s\n", coral_last_error()); ... }
 *   coral_set_input(it, 0, pixels, n);          // or write to coral_tensor_data(it, coral_input_tensor(it, 0))
 *   coral_interpreter_invoke(it);
 *   coral_get_output(it, 0, scores, m);
 *   coral_interpreter_free(it);
 *
 * Errors: functions return NULL or -1 and leave a message for coral_last_error() on the calling thread.
 * An interpreter is not thread safe; several interpreters (on one or more threads) may share a stick.
 */
#ifndef CORALRT_H
#define CORALRT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CoralInterpreter CoralInterpreter;

/* TensorFlow Lite TensorType codes, as returned by coral_tensor_type */
enum {
  CORAL_FLOAT32 = 0,
  CORAL_FLOAT16 = 1,
  CORAL_INT32 = 2,
  CORAL_UINT8 = 3,
  CORAL_INT64 = 4,
  CORAL_STRING = 5,
  CORAL_BOOL = 6,
  CORAL_INT16 = 7,
  CORAL_COMPLEX64 = 8,
  CORAL_INT8 = 9,
  CORAL_FLOAT64 = 10,
  CORAL_COMPLEX128 = 11,
  CORAL_UINT64 = 12,
  CORAL_RESOURCE = 13,
  CORAL_VARIANT = 14,
  CORAL_UINT32 = 15,
};

/* message of the last failed call on this thread (NULL if none); valid until the next failing call */
const char* coral_last_error(void);
const char* coral_version(void);

/* Edge TPUs on the bus (also ones still waiting for their firmware); -1 on error */
int32_t coral_list_devices(void);
/* copies the usb path ("bus-port.port") of stick i into buf, NUL terminated; returns the path's length */
int32_t coral_device_path(int32_t i, char* buf, size_t len);
/* 1 if stick i has no firmware yet (the first interpreter on it loads it), 0 if not */
int32_t coral_device_in_bootloader(int32_t i);

/* options: NULL, or "key=value" pairs separated by ',' or ';':
 *   device=usb | :N | usb:N | usb:<path>   the stick (default: the first); used when the model has an Edge TPU op
 *   performance=low | medium | high | max  default CORALRT_PERFORMANCE, else high (the official throttled runtime,
 *                                          250 MHz); max is the direct runtime (500 MHz, runs hotter)
 *   detection_sort=tflite2.5 | stable      how TFLite_Detection_PostProcess orders boxes of equal score: as the
 *                                          official runtime (default), or as TensorFlow Lite 2.6 and later
 * A stick already open in this process is shared at the performance it was opened with. The tensors are
 * allocated when the interpreter is made. */
CoralInterpreter* coral_interpreter_new(const char* model_path, const char* options);
CoralInterpreter* coral_interpreter_new_from_buffer(const uint8_t* model, size_t len, const char* options);
void coral_interpreter_free(CoralInterpreter* it);

/* 0 on success */
int32_t coral_interpreter_invoke(CoralInterpreter* it);
/* work out every tensor's shape again (after coral_interpreter_resize_input) and reset the variable tensors */
int32_t coral_interpreter_allocate_tensors(CoralInterpreter* it);
/* give input i the shape dims[0..rank]; tensors whose size changes move: take their coral_tensor_data again */
int32_t coral_interpreter_resize_input(CoralInterpreter* it, int32_t i, const int32_t* dims, int32_t rank);
/* variable tensors (the state of RNN / LSTM ops) back to their initial values */
int32_t coral_interpreter_reset_variables(CoralInterpreter* it);

int32_t coral_input_count(const CoralInterpreter* it);
int32_t coral_output_count(const CoralInterpreter* it);
/* tensor index of the i-th input / output, for the coral_tensor_* functions */
int32_t coral_input_tensor(const CoralInterpreter* it, int32_t i);
int32_t coral_output_tensor(const CoralInterpreter* it, int32_t i);

int32_t coral_tensor_count(const CoralInterpreter* it);
const char* coral_tensor_name(const CoralInterpreter* it, int32_t t); /* valid while the interpreter lives */
int32_t coral_tensor_type(const CoralInterpreter* it, int32_t t);
int32_t coral_tensor_rank(const CoralInterpreter* it, int32_t t);
int32_t coral_tensor_dim(const CoralInterpreter* it, int32_t t, int32_t d);
/* the shape the model allows, -1 for a dimension of any size (the shape itself if the model gives none) */
int32_t coral_tensor_signature_rank(const CoralInterpreter* it, int32_t t);
int32_t coral_tensor_signature_dim(const CoralInterpreter* it, int32_t t, int32_t d);
ptrdiff_t coral_tensor_bytes(const CoralInterpreter* it, int32_t t);
/* quantization: real = scale * (q - zero_point); count 0 = not quantized, 1 = per tensor, more = per channel */
int32_t coral_tensor_quant_count(const CoralInterpreter* it, int32_t t);
float coral_tensor_quant_scale(const CoralInterpreter* it, int32_t t, int32_t k);
int64_t coral_tensor_quant_zero_point(const CoralInterpreter* it, int32_t t, int32_t k);
int32_t coral_tensor_quantized_dimension(const CoralInterpreter* it, int32_t t);
/* the tensor's memory (8 byte aligned); valid until the tensor's size changes (resize, or an op that sizes its
 * output while it runs) */
uint8_t* coral_tensor_data(CoralInterpreter* it, int32_t t);

/* string tensors: the count, the k-th string (its length in *len, not NUL terminated; valid until the tensor
 * changes), and setting n strings data[i] of lens[i] bytes in shape dims[0..rank] (rank -1: one dimension of n) */
int32_t coral_tensor_string_count(const CoralInterpreter* it, int32_t t);
const uint8_t* coral_tensor_string(const CoralInterpreter* it, int32_t t, int32_t k, size_t* len);
int32_t coral_set_tensor_strings(CoralInterpreter* it, int32_t t, const uint8_t* const* data, const size_t* lens,
                                 int32_t n, const int32_t* dims, int32_t rank);

/* copy len bytes (the whole tensor) into input i; 0 on success */
int32_t coral_set_input(CoralInterpreter* it, int32_t i, const void* data, size_t len);
/* copy output i into data (len >= its size); returns the bytes copied */
ptrdiff_t coral_get_output(const CoralInterpreter* it, int32_t i, void* data, size_t len);

/* the ops of the main subgraph in execution order: name ("ADD" or a custom op's), and the tensor indices of its
 * inputs (io 0) or outputs (io 1), -1 for an absent optional input */
int32_t coral_op_count(const CoralInterpreter* it);
const char* coral_op_name(const CoralInterpreter* it, int32_t op);
int32_t coral_op_tensor_count(const CoralInterpreter* it, int32_t op, int32_t io);
int32_t coral_op_tensor(const CoralInterpreter* it, int32_t op, int32_t io, int32_t k);

/* the model's SignatureDefs (most Edge TPU models have none). Each maps tensor names to tensor indices, for its
 * inputs (io 0) and outputs (io 1), sorted by name as TensorFlow Lite 2.5 lists them; strings live as long as it */
int32_t coral_signature_count(const CoralInterpreter* it);
const char* coral_signature_key(const CoralInterpreter* it, int32_t s);
int32_t coral_signature_tensor_count(const CoralInterpreter* it, int32_t s, int32_t io);
const char* coral_signature_tensor_name(const CoralInterpreter* it, int32_t s, int32_t io, int32_t k);
int32_t coral_signature_tensor_index(const CoralInterpreter* it, int32_t s, int32_t io, int32_t k);

/* usb path of the stick the interpreter runs on ("" if the model has no Edge TPU op); returns its length */
int32_t coral_interpreter_device_path(const CoralInterpreter* it, char* buf, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* CORALRT_H */
