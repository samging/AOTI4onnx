//! A loaded model: opening a `.pt2`, running it, and freeing it.
//!
//! Opening goes in four steps. The `.pt2` is a zip archive, so its `*.wrapper.so`
//! is found and copied to a temporary file. That file is `dlopen`ed, and the
//! temporary copy is deleted straight away, since the loaded library no longer
//! needs it on disk. The library's functions are looked up ([`crate::api`]), and a
//! container with `num_models` instances is created from them.
//!
//! Running goes in three steps. The caller's input is wrapped as a libtorch tensor
//! without copying it. The container runs with that input and returns output
//! tensors. The outputs are read ([`crate::tensor`]) and then released. The
//! variants differ only in what they do with the outputs: copy them into a new
//! `Vec` ([`AotiModel::run`]), copy them into a buffer the caller owns
//! ([`AotiModel::run_into`]), or borrow libtorch's memory ([`AotiModel::run_view`]).

use std::ffi::{c_void, CString};
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::process;
use std::ptr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use libloading::Library;

use crate::abi::{Handle, LAYOUT_STRIDED};
use crate::api::{check, Api};
use crate::error::Error;
use crate::output::OutputView;
use crate::shape::{checked_numel, row_major_strides};
use crate::tensor::{inspect, Tensor};

/// The loaded library and its function table, kept together. An output view holds
/// a clone of this `Arc`, so the library stays loaded while any view is alive.
pub(crate) struct Shared {
    pub(crate) api: Api,
    pub(crate) _lib: Library,
}

/// A loaded AOTInductor model. Dropping it deletes its container.
///
/// The container is created with `num_models` instances. Each instance can run one
/// call at a time, and calls to different instances run concurrently. This type is
/// `Send` and `Sync`, so one model can be shared between threads.
pub struct AotiModel {
    pub(crate) shared: Arc<Shared>,
    container: Handle,
}

// Sound because the AOTInductor container accepts concurrent `Run` calls, one per
// model instance (`num_models`). Each call uses its own input and output handles.
unsafe impl Send for AotiModel {}
unsafe impl Sync for AotiModel {}

impl AotiModel {
    /// Loads a `.pt2` package with `num_models` instances that can run concurrently.
    ///
    /// Fails with [`Error::InvalidRunners`] when `num_models` is 0, and with the
    /// errors of extracting, loading or resolving the library otherwise.
    pub fn load(path: impl AsRef<Path>, num_models: usize) -> Result<Self, Error> {
        open(path.as_ref(), num_models)
    }

    /// Runs the model and copies every output into a new `Vec`.
    ///
    /// `input` is read in place, not copied; `shape` says how its elements are laid out.
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
    /// borrows libtorch's output memory, so nothing is copied; the memory is
    /// released when the view is dropped.
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

/// Extracts the package's library, loads it and creates the container. Fails with
/// [`Error::InvalidRunners`] for 0 instances.
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

/// Copies the package's `*.wrapper.so` to a uniquely named file in the temporary
/// directory, and returns its path. The name includes the process id and a
/// timestamp, so two processes opening models at once don't collide.
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

/// The one output of a model, or [`Error::OutputCount`] when it has another number.
fn single_output(outputs: &[Handle]) -> Result<Handle, Error> {
    match outputs {
        [only] => Ok(*only),
        other => Err(Error::OutputCount(other.len())),
    }
}

/// Releases output tensors the caller has finished with.
fn release(api: &Api, handles: &[Handle]) {
    for &h in handles {
        unsafe { (api.delete_tensor)(h) };
    }
}
