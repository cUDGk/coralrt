/* coralrt: runs Coral Edge TPU models (*_edgetpu.tflite) on the Coral USB Accelerator without libedgetpu
 * or TensorFlow Lite. Link against coralrt.dll / libcoralrt.so / libcoralrt.dylib (built with
 * `cargo build --release`). The stick is reached through libusb-1.0, loaded at run time (on Windows the
 * libusb-1.0.dll and WinUSB driver that the official Edge TPU runtime installs; CORALRT_LIBUSB overrides
 * the library path).
 *
 *   CoralInterpreter* it = coral_interpreter_new("model_edgetpu.tflite", NULL, NULL);
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
  CORAL_BOOL = 6,
  CORAL_INT16 = 7,
  CORAL_INT8 = 9,
  CORAL_FLOAT64 = 10,
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

/* device: NULL / "usb" for the first stick, ":N" / "usb:N" for the N-th, "usb:<path>" for a given one.
 * performance: NULL (CORALRT_PERFORMANCE, else "high") or "low", "medium", "high" (the official throttled
 * runtime, 250 MHz), "max" (the direct runtime, 500 MHz; runs hotter). A stick already open in this process
 * is shared at the performance it was opened with. */
CoralInterpreter* coral_interpreter_new(const char* model_path, const char* device, const char* performance);
CoralInterpreter* coral_interpreter_new_from_buffer(const uint8_t* model, size_t len, const char* device,
                                                    const char* performance);
void coral_interpreter_free(CoralInterpreter* it);

/* 0 on success */
int32_t coral_interpreter_invoke(CoralInterpreter* it);

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
ptrdiff_t coral_tensor_bytes(const CoralInterpreter* it, int32_t t);
/* quantization: real = scale * (q - zero_point); count 0 = not quantized, 1 = per tensor, more = per channel */
int32_t coral_tensor_quant_count(const CoralInterpreter* it, int32_t t);
float coral_tensor_quant_scale(const CoralInterpreter* it, int32_t t, int32_t k);
int64_t coral_tensor_quant_zero_point(const CoralInterpreter* it, int32_t t, int32_t k);
int32_t coral_tensor_quantized_dimension(const CoralInterpreter* it, int32_t t);
/* the tensor's memory (8 byte aligned), valid while the interpreter lives */
uint8_t* coral_tensor_data(CoralInterpreter* it, int32_t t);

/* copy len bytes (the whole tensor) into input i; 0 on success */
int32_t coral_set_input(CoralInterpreter* it, int32_t i, const void* data, size_t len);
/* copy output i into data (len >= its size); returns the bytes copied */
ptrdiff_t coral_get_output(const CoralInterpreter* it, int32_t i, void* data, size_t len);

/* usb path of the stick the interpreter runs on ("" if the model has no Edge TPU op); returns its length */
int32_t coral_interpreter_device_path(const CoralInterpreter* it, char* buf, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* CORALRT_H */
