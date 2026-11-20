//! Pure-Rust loader for AOTInductor `.pt2` packages.
//!
//! A `.pt2` file is a zip archive that contains a compiled `*.wrapper.so`.
//! `AotiModel::load` extracts that library, loads it with `dlopen`, and calls
//! the exported `AOTInductorModelContainer*` and `aoti_torch_*` C functions.
//! libtorch must be linked into the final binary (see `build.rs`).
//!
//! Inputs are wrapped around the caller's memory without a copy. Outputs are
//! float32 and must be contiguous. Each `AotiModel` owns its own container.

pub mod ffi;

use std::ffi::{c_char, c_void, CString};
use std::fmt;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::process;
use std::ptr;
use std::slice;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use libloading::os::unix::Library as UnixLibrary;
use libloading::Library;

type Handle = *mut c_void;

type CreateFn = unsafe extern "C" fn(*mut Handle, usize, *const c_char, *const c_char) -> i32;
type DeleteContainerFn = unsafe extern "C" fn(Handle) -> i32;
type RunFn = unsafe extern "C" fn(Handle, *mut Handle, usize, *mut Handle, usize, *mut c_void, *mut c_void) -> i32;
type NumOutputsFn = unsafe extern "C" fn(Handle, *mut usize) -> i32;
type CreateBlobFn = unsafe extern "C" fn(
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
type DataPtrFn = unsafe extern "C" fn(Handle, *mut *mut c_void) -> i32;
type DimFn = unsafe extern "C" fn(Handle, *mut i64) -> i32;
type SizesFn = unsafe extern "C" fn(Handle, *mut *mut i64) -> i32;
type StridesFn = unsafe extern "C" fn(Handle, *mut *mut i64) -> i32;
type DtypeFn = unsafe extern "C" fn(Handle, *mut i32) -> i32;
type DeleteTensorFn = unsafe extern "C" fn(Handle) -> i32;
type ConstI32Fn = unsafe extern "C" fn() -> i32;

const SUCCESS: i32 = 0;
const LAYOUT_STRIDED: i32 = 0;

#[derive(Debug)]
pub enum Error {
    InvalidPath,
    Io(io::Error),
    Zip(zip::result::ZipError),
    NoWrapperInPackage,
    LoadLibrary(libloading::Error),
    MissingSymbol { name: &'static str, reason: String },
    Aoti { call: &'static str, code: i32 },
    InvalidShape(String),
    ShapeMismatch { expected: usize, got: usize },
    OutputSizeMismatch { expected: usize, got: usize },
    OutputCount(usize),
    UnsupportedDtype(i32),
    NonContiguous,
    InvalidRunners,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidPath => write!(f, "model path is not valid UTF-8"),
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Zip(e) => write!(f, "invalid .pt2 archive: {e}"),
            Error::NoWrapperInPackage => write!(f, "no *.wrapper.so found in the package"),
            Error::LoadLibrary(e) => write!(f, "failed to load the compiled model library: {e}"),
            Error::MissingSymbol { name, reason } => write!(f, "missing symbol {name}: {reason}"),
            Error::Aoti { call, code } => write!(f, "{call} failed with code {code}"),
            Error::InvalidShape(msg) => write!(f, "invalid shape: {msg}"),
            Error::ShapeMismatch { expected, got } => {
                write!(f, "input has {got} elements, shape implies {expected}")
            }
            Error::OutputSizeMismatch { expected, got } => {
                write!(f, "output has {expected} elements, buffer holds {got}")
            }
            Error::OutputCount(n) => write!(f, "model returned {n} outputs, expected 1"),
            Error::UnsupportedDtype(code) => write!(f, "only float32 outputs are supported (dtype code {code})"),
            Error::NonContiguous => write!(f, "output tensor is not contiguous"),
            Error::InvalidRunners => write!(f, "num_runners must be at least 1"),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<zip::result::ZipError> for Error {
    fn from(e: zip::result::ZipError) -> Self {
        Error::Zip(e)
    }
}

/// A float32 output tensor, copied out of libtorch.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
}

struct Api {
    create: CreateFn,
    delete_container: DeleteContainerFn,
    run: RunFn,
    num_outputs: NumOutputsFn,
    create_blob: CreateBlobFn,
    data_ptr: DataPtrFn,
    dim: DimFn,
    sizes: SizesFn,
    strides: StridesFn,
    dtype: DtypeFn,
    delete_tensor: DeleteTensorFn,
    dtype_float32: ConstI32Fn,
    device_cpu: ConstI32Fn,
}

impl Api {
    /// `AOTInductor*` symbols come from the model library; the `aoti_torch_*`
    /// helpers are exported by libtorch itself, so they are looked up globally.
    fn load(lib: &Library) -> Result<Api, Error> {
        let global = UnixLibrary::this();
        Ok(Api {
            create: sym(lib, "AOTInductorModelContainerCreateWithDevice")?,
            delete_container: sym(lib, "AOTInductorModelContainerDelete")?,
            run: sym(lib, "AOTInductorModelContainerRun")?,
            num_outputs: sym(lib, "AOTInductorModelContainerGetNumOutputs")?,
            create_blob: global_sym(&global, "aoti_torch_create_tensor_from_blob_v2")?,
            data_ptr: global_sym(&global, "aoti_torch_get_data_ptr")?,
            dim: global_sym(&global, "aoti_torch_get_dim")?,
            sizes: global_sym(&global, "aoti_torch_get_sizes")?,
            strides: global_sym(&global, "aoti_torch_get_strides")?,
            dtype: global_sym(&global, "aoti_torch_get_dtype")?,
            delete_tensor: global_sym(&global, "aoti_torch_delete_tensor_object")?,
            dtype_float32: global_sym(&global, "aoti_torch_dtype_float32")?,
            device_cpu: global_sym(&global, "aoti_torch_device_type_cpu")?,
        })
    }
}

fn sym<T: Copy>(lib: &Library, name: &'static str) -> Result<T, Error> {
    let c_name = CString::new(name).expect("symbol names contain no NUL");
    let symbol = unsafe { lib.get::<T>(c_name.as_bytes_with_nul()) }.map_err(|e| Error::MissingSymbol {
        name,
        reason: e.to_string(),
    })?;
    Ok(*symbol)
}

fn global_sym<T: Copy>(lib: &UnixLibrary, name: &'static str) -> Result<T, Error> {
    let c_name = CString::new(name).expect("symbol names contain no NUL");
    let symbol = unsafe { lib.get::<T>(c_name.as_bytes_with_nul()) }.map_err(|e| Error::MissingSymbol {
        name,
        reason: e.to_string(),
    })?;
    Ok(*symbol)
}

fn check(call: &'static str, code: i32) -> Result<(), Error> {
    if code == SUCCESS {
        Ok(())
    } else {
        Err(Error::Aoti { call, code })
    }
}

/// Library and function pointers. Shared with output views so the library
/// stays loaded while a view is alive.
struct Shared {
    api: Api,
    _lib: Library,
}

/// Handle to a loaded AOTInductor model. Dropping it frees the container.
pub struct AotiModel {
    shared: Arc<Shared>,
    container: Handle,
}

// Sound because the AOTInductor container accepts concurrent `Run` calls, one per
// model instance (`num_models`). Each call uses its own input and output handles.
unsafe impl Send for AotiModel {}
unsafe impl Sync for AotiModel {}

impl AotiModel {
    /// Loads a `.pt2` package with `num_models` instances that can run concurrently.
    pub fn load(path: impl AsRef<Path>, num_models: usize) -> Result<Self, Error> {
        open(path.as_ref(), num_models)
    }

    /// Runs the model and copies every output into a new `Vec`.
    pub fn run(&self, input: &[f32], shape: &[i64]) -> Result<Vec<Tensor>, Error> {
        let outputs = self.execute(input, shape)?;
        let shared = self.shared.clone();
        let result = outputs
            .iter()
            .map(|&h| unsafe { inspect(&shared.api, h).map(|info| info.to_tensor()) })
            .collect();
        release(&shared.api, &outputs);
        result
    }

    /// Runs a single-output model and writes the result into `out`, which must
    /// hold exactly the output's element count. No allocation for the output.
    pub fn run_into(&self, input: &[f32], shape: &[i64], out: &mut [f32]) -> Result<(), Error> {
        let outputs = self.execute(input, shape)?;
        let shared = self.shared.clone();
        let result = (|| {
            let handle = single_output(&outputs)?;
            let info = unsafe { inspect(&shared.api, handle) }?;
            if info.numel != out.len() {
                return Err(Error::OutputSizeMismatch { expected: info.numel, got: out.len() });
            }
            out.copy_from_slice(unsafe { info.data() });
            Ok(())
        })();
        release(&shared.api, &outputs);
        result
    }

    /// Runs a single-output model and returns a view of its data. The view
    /// borrows libtorch's output memory, so nothing is copied.
    pub fn run_view(&self, input: &[f32], shape: &[i64]) -> Result<OutputView, Error> {
        let outputs = self.execute(input, shape)?;
        let shared = self.shared.clone();
        let result = single_output(&outputs).and_then(|handle| unsafe { inspect(&shared.api, handle) });
        match result {
            Ok(info) => Ok(OutputView { shared, handle: outputs[0], info }),
            Err(e) => {
                release(&shared.api, &outputs);
                Err(e)
            }
        }
    }

    /// Wraps the caller's input memory as a tensor and runs the container.
    /// Returns the output handles, which the caller must release.
    fn execute(&self, input: &[f32], shape: &[i64]) -> Result<Vec<Handle>, Error> {
        let numel = checked_numel(shape)?;
        if numel != input.len() {
            return Err(Error::ShapeMismatch { expected: numel, got: input.len() });
        }
        let api = &self.shared.api;
        let strides = row_major_strides(shape);
        let dtype = unsafe { (api.dtype_float32)() };
        let device = unsafe { (api.device_cpu)() };

        let mut num_outputs: usize = 0;
        check("AOTInductorModelContainerGetNumOutputs", unsafe {
            (api.num_outputs)(self.container, &mut num_outputs)
        })?;

        // The input tensor borrows `input`. The container takes ownership of the
        // handle when it runs, so it is only deleted here if the run never starts.
        let mut input_handle: Handle = ptr::null_mut();
        check("aoti_torch_create_tensor_from_blob_v2", unsafe {
            (api.create_blob)(
                input.as_ptr() as *mut c_void,
                shape.len() as i64,
                shape.as_ptr(),
                strides.as_ptr(),
                0,
                dtype,
                device,
                0,
                &mut input_handle,
                LAYOUT_STRIDED,
                ptr::null(),
                0,
            )
        })?;

        let mut inputs = [input_handle];
        let mut outputs: Vec<Handle> = vec![ptr::null_mut(); num_outputs];
        check("AOTInductorModelContainerRun", unsafe {
            (api.run)(
                self.container,
                inputs.as_mut_ptr(),
                1,
                outputs.as_mut_ptr(),
                num_outputs,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        })?;
        Ok(outputs)
    }
}

impl Drop for AotiModel {
    fn drop(&mut self) {
        if !self.container.is_null() {
            unsafe { (self.shared.api.delete_container)(self.container) };
        }
    }
}

/// A single output tensor that stays in libtorch's memory until dropped.
pub struct OutputView {
    shared: Arc<Shared>,
    handle: Handle,
    info: TensorInfo,
}

impl OutputView {
    pub fn shape(&self) -> &[i64] {
        unsafe { self.info.shape() }
    }

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

fn open(package: &Path, num_models: usize) -> Result<AotiModel, Error> {
    if num_models == 0 {
        return Err(Error::InvalidRunners);
    }
    let so = extract_wrapper(package)?;
    let loaded = unsafe { Library::new(&so) };
    let _ = fs::remove_file(&so);
    let lib = loaded.map_err(Error::LoadLibrary)?;

    let api = Api::load(&lib)?;
    let device = CString::new("cpu").expect("no NUL in device name");
    let mut container: Handle = ptr::null_mut();
    check("AOTInductorModelContainerCreateWithDevice", unsafe {
        (api.create)(&mut container, num_models, device.as_ptr(), ptr::null())
    })?;

    Ok(AotiModel {
        shared: Arc::new(Shared { api, _lib: lib }),
        container,
    })
}

fn extract_wrapper(package: &Path) -> Result<PathBuf, Error> {
    let mut archive = zip::ZipArchive::new(File::open(package)?)?;
    let name = archive
        .file_names()
        .find(|n| n.ends_with(".wrapper.so"))
        .map(str::to_owned)
        .ok_or(Error::NoWrapperInPackage)?;

    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    let out = std::env::temp_dir().join(format!("aoti_ffi_{}_{}.so", process::id(), nanos));
    let mut entry = archive.by_name(&name)?;
    let mut file = File::create(&out)?;
    io::copy(&mut entry, &mut file)?;
    Ok(out)
}

fn single_output(outputs: &[Handle]) -> Result<Handle, Error> {
    match outputs {
        [only] => Ok(*only),
        other => Err(Error::OutputCount(other.len())),
    }
}

fn release(api: &Api, handles: &[Handle]) {
    for &h in handles {
        unsafe { (api.delete_tensor)(h) };
    }
}

fn checked_numel(shape: &[i64]) -> Result<usize, Error> {
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

fn row_major_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Pointers into a libtorch output tensor. Valid while the tensor is alive.
#[derive(Clone, Copy)]
struct TensorInfo {
    shape_ptr: *const i64,
    ndim: usize,
    data_ptr: *const f32,
    numel: usize,
}

impl TensorInfo {
    unsafe fn shape<'a>(&self) -> &'a [i64] {
        if self.ndim == 0 {
            &[]
        } else {
            slice::from_raw_parts(self.shape_ptr, self.ndim)
        }
    }

    unsafe fn data<'a>(&self) -> &'a [f32] {
        if self.numel == 0 {
            &[]
        } else {
            slice::from_raw_parts(self.data_ptr, self.numel)
        }
    }

    unsafe fn to_tensor(&self) -> Tensor {
        Tensor {
            shape: self.shape().to_vec(),
            data: self.data().to_vec(),
        }
    }
}

/// Checks dtype and contiguity of an output tensor and returns its pointers.
unsafe fn inspect(api: &Api, handle: Handle) -> Result<TensorInfo, Error> {
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
