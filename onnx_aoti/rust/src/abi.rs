//! The C signatures of the functions this crate calls, and the constants they take.
//!
//! Two sets of functions. The `AOTInductor*` ones are exported by the model's own
//! `*.wrapper.so`: they create a container of model instances, run it, and report
//! its outputs. The `aoti_torch_*` ones are exported by libtorch: they create,
//! inspect and free the tensors passed in and out. Each type alias below is the
//! `extern "C"` signature of one of them, written as it is in the C headers.
//!
//! Tensors and containers are opaque to Rust: a [`Handle`] is a pointer to a
//! libtorch object, never dereferenced here.

use std::ffi::{c_char, c_void};

/// An opaque pointer to a libtorch tensor or to an AOTInductor container.
pub type Handle = *mut c_void;

/// `AOTInductorModelContainerCreateWithDevice(container, num_models, device, config)`.
pub type CreateFn = unsafe extern "C" fn(*mut Handle, usize, *const c_char, *const c_char) -> i32;

/// `AOTInductorModelContainerDelete(container)`.
pub type DeleteContainerFn = unsafe extern "C" fn(Handle) -> i32;

/// `AOTInductorModelContainerRun(container, inputs, n_in, outputs, n_out, stream, proxy)`.
/// The last two are a CUDA stream and a proxy executor; both are null on the CPU.
pub type RunFn = unsafe extern "C" fn(Handle, *mut Handle, usize, *mut Handle, usize, *mut c_void, *mut c_void) -> i32;

/// `AOTInductorModelContainerGetNumOutputs(container, out)`.
pub type NumOutputsFn = unsafe extern "C" fn(Handle, *mut usize) -> i32;

/// `aoti_torch_create_tensor_from_blob_v2`: wraps existing memory as a tensor, with no copy.
/// The arguments are the data pointer, the rank, the sizes, the strides, the storage offset,
/// the dtype, the device type, the device index, the output handle, the layout, and the
/// optional opaque data and its length.
pub type CreateBlobFn = unsafe extern "C" fn(
    *mut c_void,
    i64,
    *const i64,
    *const i64,
    i64,
    i32,
    i32,
    i32,
    *mut Handle,
    i32,
    *const u8,
    i64,
) -> i32;

/// `aoti_torch_get_data_ptr(tensor, out)`: the address of the tensor's first element.
pub type DataPtrFn = unsafe extern "C" fn(Handle, *mut *mut c_void) -> i32;

/// `aoti_torch_get_dim(tensor, out)`: the number of dimensions.
pub type DimFn = unsafe extern "C" fn(Handle, *mut i64) -> i32;

/// `aoti_torch_get_sizes(tensor, out)`: a pointer to the sizes, owned by libtorch.
pub type SizesFn = unsafe extern "C" fn(Handle, *mut *mut i64) -> i32;

/// `aoti_torch_get_strides(tensor, out)`: a pointer to the strides, owned by libtorch.
pub type StridesFn = unsafe extern "C" fn(Handle, *mut *mut i64) -> i32;

/// `aoti_torch_get_dtype(tensor, out)`: the dtype as libtorch's number for it.
pub type DtypeFn = unsafe extern "C" fn(Handle, *mut i32) -> i32;

/// `aoti_torch_delete_tensor_object(tensor)`: releases a tensor handle.
pub type DeleteTensorFn = unsafe extern "C" fn(Handle) -> i32;

/// A libtorch function that returns a constant, such as the float32 dtype's number.
pub type ConstI32Fn = unsafe extern "C" fn() -> i32;

/// The status every call returns when it succeeded.
pub const SUCCESS: i32 = 0;

/// `aoti_torch` layout code for strided (ordinary) tensors, the only layout this crate makes.
pub const LAYOUT_STRIDED: i32 = 0;
