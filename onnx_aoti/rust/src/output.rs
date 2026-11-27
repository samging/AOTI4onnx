//! A borrowed output: the result of a run, left in libtorch's memory.
//!
//! [`AotiModel::run_view`](crate::AotiModel::run_view) returns one of these instead
//! of copying. The data and shape are read straight from libtorch's tensor, which
//! is released when the view is dropped. The view also holds a clone of the model's
//! shared library, so the code that owns the memory stays loaded for as long as the
//! view exists, even if the model itself is dropped first.

use std::sync::Arc;

use crate::abi::Handle;
use crate::model::Shared;
use crate::tensor::TensorInfo;

/// A single output tensor that stays in libtorch's memory until dropped.
pub struct OutputView {
    pub(crate) shared: Arc<Shared>,
    pub(crate) handle: Handle,
    pub(crate) info: TensorInfo,
}

impl OutputView {
    /// The sizes of the output, such as `[1, 3]`.
    pub fn shape(&self) -> &[i64] {
        unsafe { self.info.shape() }
    }

    /// The output's elements in row-major order, borrowed from libtorch.
    pub fn data(&self) -> &[f32] {
        unsafe { self.info.data() }
    }

    /// Raw pointers for FFI: (data, numel, shape, rank).
    pub(crate) fn raw_parts(&self) -> (*const f32, usize, *const i64, usize) {
        (self.info.data_ptr, self.info.numel, self.info.shape_ptr, self.info.ndim)
    }
}

impl Drop for OutputView {
    fn drop(&mut self) {
        unsafe { (self.shared.api.delete_tensor)(self.handle) };
    }
}
