//! Shape arithmetic for the caller's input and for checking outputs.
//!
//! A tensor's shape gives its dimensions, such as `[1, 3, 224, 224]`. Its strides
//! give how far apart neighbouring elements are in memory. Row-major (C order)
//! means the last dimension varies fastest, so the stride of a dimension is the
//! product of all the sizes after it.

use crate::error::Error;

/// The number of elements a shape holds, checking that it is non-empty, that every
/// dimension is positive, and that the product fits in a `usize`.
pub(crate) fn checked_numel(shape: &[i64]) -> Result<usize, Error> {
    if shape.is_empty() {
        return Err(Error::InvalidShape("shape is empty".into()));
    }
    let mut numel: usize = 1;
    for &d in shape {
        if d <= 0 {
            return Err(Error::InvalidShape(format!("dimension {d} is not positive")));
        }
        numel = numel
            .checked_mul(d as usize)
            .ok_or_else(|| Error::InvalidShape("shape is too large".into()))?;
    }
    Ok(numel)
}

/// The row-major strides of a shape, in elements: `[3, 4]` gives `[4, 1]`.
pub(crate) fn row_major_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
