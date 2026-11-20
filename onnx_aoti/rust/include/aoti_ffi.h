#ifndef AOTI_FFI_H
#define AOTI_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AotiModel AotiModel;

/* Output of aoti_run_view. Release raw_handle with aoti_result_free. */
typedef struct AotiOutput {
    const float* data_ptr;   /* element data, valid until aoti_result_free */
    size_t data_len;         /* number of float32 elements */
    const int64_t* shape_ptr;
    size_t shape_len;        /* rank */
    void* raw_handle;
} AotiOutput;

/* Loads a .pt2 package. num_runners (>= 1) is the number of instances that
 * can run concurrently. Returns NULL on failure (see aoti_last_error). */
AotiModel* aoti_load(const char* path, size_t num_runners);

/* Releases the reference from aoti_load. Calls still running keep the model alive. */
void aoti_free(AotiModel* model);

/* Runs the model. input holds input_len float32 values laid out as shape.
 * Writes the single output into out, which must hold exactly its element
 * count. Returns 0, or -1 on error. */
int32_t aoti_run_into(const AotiModel* model, const float* input, size_t input_len,
                      const int64_t* shape, size_t ndim,
                      float* out, size_t out_len);

/* Runs the model and returns a view of its single output. Returns 0, or -1 on error. */
int32_t aoti_run_view(const AotiModel* model, const float* input, size_t input_len,
                      const int64_t* shape, size_t ndim,
                      AotiOutput* out);

/* Releases the output from aoti_run_view. */
void aoti_result_free(void* raw_handle);

/* Copies this thread's last error as a NUL-terminated string. Returns the
 * message length in bytes, excluding the terminator. */
size_t aoti_last_error(char* buf, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* AOTI_FFI_H */
