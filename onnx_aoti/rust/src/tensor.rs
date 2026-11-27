//! Output tensors: the copied result type, and how a libtorch output is read.
//!
//! libtorch owns the memory of an output tensor. Reading one means asking libtorch
//! for its dtype, rank, sizes, strides and data address, and checking the layout
//! before any element is read. [`inspect`] does that and returns a [`TensorInfo`]:
//! raw pointers that stay valid only while the output tensor is alive.

use std::ffi::c_void;
use std::ptr;
use std::slice;

use crate::abi::Handle;
use crate::api::{check, Api};
use crate::error::Error;
use crate::shape::row_major_strides;

/// A float32 output tensor, copied out of libtorch so it outlives the call.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
}

/// Pointers into a libtorch output tensor. Valid while the tensor is alive.
#[derive(Clone, Copy)]
pub(crate) struct TensorInfo {
    pub(crate) shape_ptr: *const i64,
    pub(crate) ndim: usize,
    pub(crate) data_ptr: *const f32,
    pub(crate) numel: usize,
}

impl TensorInfo {
    /// The sizes of the tensor, as libtorch holds them.
    ///
    /// # Safety
    /// The output tensor this came from must still be alive.
    pub(crate) unsafe fn shape<'a>(&self) -> &'a [i64] {
        if self.ndim == 0 {
            &[]
        } else {
            slice::from_raw_parts(self.shape_ptr, self.ndim)
        }
    }

    /// The elements of the tensor, as libtorch holds them.
    ///
    /// # Safety
    /// The output tensor this came from must still be alive.
    pub(crate) unsafe fn data<'a>(&self) -> &'a [f32] {
        if self.numel == 0 {
            &[]
        } else {
            slice::from_raw_parts(self.data_ptr, self.numel)
        }
    }

    /// Copies the tensor out of libtorch.
    ///
    /// # Safety
    /// The output tensor this came from must still be alive.
    pub(crate) unsafe fn to_tensor(&self) -> Tensor {
        Tensor {
            shape: self.shape().to_vec(),
            data: self.data().to_vec(),
        }
    }
}

/// Checks the dtype and layout of an output tensor and returns pointers to its data.
///
/// Only float32 is accepted, and only tensors whose strides are row-major, since
/// [`TensorInfo::data`] reads elements as one contiguous run. A size-1 dimension
/// may carry any stride, because it never moves the element index.
///
/// # Safety
/// `handle` must be a live libtorch tensor. The returned pointers are valid only
/// while it stays alive.
pub(crate) unsafe fn inspect(api: &Api, handle: Handle) -> Result<TensorInfo, Error> {
    let mut dtype: i32 = 0;
    check("aoti_torch_get_dtype", (api.dtype)(handle, &mut dtype))?;
    if dtype != (api.dtype_float32)() {
        return Err(Error::UnsupportedDtype(dtype));
    }

    let mut ndim: i64 = 0;
    check("aoti_torch_get_dim", (api.dim)(handle, &mut ndim))?;
    let ndim = ndim as usize;

    let (shape_ptr, strides_ptr) = if ndim == 0 {
        (ptr::null(), ptr::null())
    } else {
        let mut sizes: *mut i64 = ptr::null_mut();
        check("aoti_torch_get_sizes", (api.sizes)(handle, &mut sizes))?;
        let mut strides: *mut i64 = ptr::null_mut();
        check("aoti_torch_get_strides", (api.strides)(handle, &mut strides))?;
        (sizes as *const i64, strides as *const i64)
    };

    let info_shape = if ndim == 0 { &[][..] } else { slice::from_raw_parts(shape_ptr, ndim) };
    let info_strides = if ndim == 0 { &[][..] } else { slice::from_raw_parts(strides_ptr, ndim) };
    let expected = row_major_strides(info_shape);
    for i in 0..ndim {
        // Size-1 dimensions can carry any stride.
        if info_shape[i] != 1 && info_strides[i] != expected[i] {
            return Err(Error::NonContiguous);
        }
    }

    let numel = info_shape.iter().product::<i64>() as usize;
    let mut data_ptr: *mut c_void = ptr::null_mut();
    if numel > 0 {
        check("aoti_torch_get_data_ptr", (api.data_ptr)(handle, &mut data_ptr))?;
    }

    Ok(TensorInfo {
        shape_ptr,
        ndim,
        data_ptr: data_ptr as *const f32,
        numel,
    })
}
