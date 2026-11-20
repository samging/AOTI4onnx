//! C ABI for calling AOTI models from other languages (Kotlin/JVM via FFM).
//! The declarations are in `include/aoti_ffi.h`.
//!
//! Ownership:
//! - `aoti_load` returns a model. Release it with `aoti_free`. Calls that are
//!   still running keep the model alive until they return.
//! - `aoti_run_view` fills an `AotiOutput`. Release its `raw_handle` with
//!   `aoti_result_free`. The data pointers are valid until then.
//! - `aoti_run_into` writes into a caller-owned buffer and needs no release.
//!
//! Errors: functions return NULL or a negative status. The message is stored
//! per thread and read with `aoti_last_error`.

use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
use std::slice;
use std::sync::Arc;

use crate::{AotiModel, OutputView};

/// Output of `aoti_run_view`. Layout matches `AotiOutput` in the header.
#[repr(C)]
pub struct AotiOutput {
    pub data_ptr: *const f32,
    pub data_len: usize,
    pub shape_ptr: *const i64,
    pub shape_len: usize,
    pub raw_handle: *mut c_void,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error(message: impl Into<String>) {
    let message = message.into().replace('\0', "?");
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(CString::new(message).expect("NULs were removed")));
}

/// Runs `f`, turning errors and panics into `fallback` plus a stored message.
/// Panics must not unwind across the C ABI.
fn guarded<T>(fallback: T, f: impl FnOnce() -> Result<T, String>) -> T {
    match panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => value,
        Ok(Err(message)) => {
            set_error(message);
            fallback
        }
        Err(_) => {
            set_error("panic inside aoti_ffi");
            fallback
        }
    }
}

unsafe fn c_str<'a>(p: *const c_char) -> Result<&'a str, String> {
    if p.is_null() {
        return Err("null path".into());
    }
    CStr::from_ptr(p).to_str().map_err(|e| e.to_string())
}

unsafe fn input_slice<'a, T>(p: *const T, len: usize, name: &str) -> Result<&'a [T], String> {
    if len == 0 {
        return Ok(&[]);
    }
    if p.is_null() {
        return Err(format!("null {name}"));
    }
    Ok(slice::from_raw_parts(p, len))
}

unsafe fn output_slice<'a, T>(p: *mut T, len: usize, name: &str) -> Result<&'a mut [T], String> {
    if len == 0 {
        return Ok(&mut []);
    }
    if p.is_null() {
        return Err(format!("null {name}"));
    }
    Ok(slice::from_raw_parts_mut(p, len))
}

/// Takes a counted reference to the model. The count is held for the duration
/// of the call, so `aoti_free` cannot free the model underneath it.
unsafe fn model_ref(p: *const AotiModel) -> Result<Arc<AotiModel>, String> {
    if p.is_null() {
        return Err("null model".into());
    }
    Arc::increment_strong_count(p);
    Ok(Arc::from_raw(p))
}

/// Loads a `.pt2` package. `num_runners` is the number of instances that can run concurrently.
#[no_mangle]
pub unsafe extern "C" fn aoti_load(path: *const c_char, num_runners: usize) -> *const AotiModel {
    guarded(ptr::null(), || {
        let path = c_str(path)?;
        let model = AotiModel::load(path, num_runners).map_err(|e| e.to_string())?;
        Ok(Arc::into_raw(Arc::new(model)))
    })
}

/// Releases the caller's reference. The model is freed when the last call using it returns.
#[no_mangle]
pub unsafe extern "C" fn aoti_free(model: *const AotiModel) {
    if !model.is_null() {
        drop(Arc::from_raw(model));
    }
}

/// Runs the model and writes its single output into `out`, which must hold
/// exactly the output's element count. Returns 0 on success or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn aoti_run_into(
    model: *const AotiModel,
    input: *const f32,
    input_len: usize,
    shape: *const i64,
    ndim: usize,
    out: *mut f32,
    out_len: usize,
) -> i32 {
    guarded(-1, || {
        let model = model_ref(model)?;
        let input = input_slice(input, input_len, "input")?;
        let shape = input_slice(shape, ndim, "shape")?;
        let out = output_slice(out, out_len, "output")?;
        model.run_into(input, shape, out).map_err(|e| e.to_string())?;
        Ok(0)
    })
}

/// Runs the model and returns a view of its single output in `out`. Release
/// `out->raw_handle` with `aoti_result_free`. Returns 0 on success or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn aoti_run_view(
    model: *const AotiModel,
    input: *const f32,
    input_len: usize,
    shape: *const i64,
    ndim: usize,
    out: *mut AotiOutput,
) -> i32 {
    guarded(-1, || {
        if out.is_null() {
            return Err("null output".into());
        }
        let model = model_ref(model)?;
        let input = input_slice(input, input_len, "input")?;
        let shape = input_slice(shape, ndim, "shape")?;
        let view: OutputView = model.run_view(input, shape).map_err(|e| e.to_string())?;
        let (data_ptr, data_len, shape_ptr, shape_len) = view.raw_parts();
        *out = AotiOutput {
            data_ptr,
            data_len,
            shape_ptr,
            shape_len,
            raw_handle: Box::into_raw(Box::new(view)) as *mut c_void,
        };
        Ok(0)
    })
}

#[no_mangle]
pub unsafe extern "C" fn aoti_result_free(raw_handle: *mut c_void) {
    if !raw_handle.is_null() {
        drop(Box::from_raw(raw_handle as *mut OutputView));
    }
}

/// Copies this thread's last error message into `buf` as a NUL-terminated
/// string. Returns the full message length in bytes, excluding the terminator.
#[no_mangle]
pub unsafe extern "C" fn aoti_last_error(buf: *mut c_char, cap: usize) -> usize {
    LAST_ERROR.with(|e| {
        let e = e.borrow();
        let bytes: &[u8] = match e.as_ref() {
            Some(message) => message.as_bytes(),
            None => &[],
        };
        if !buf.is_null() && cap > 0 {
            let n = bytes.len().min(cap - 1);
            ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, n);
            *buf.add(n) = 0;
        }
        bytes.len()
    })
}
