//! The table of C functions a loaded model uses, and how each one is found.
//!
//! The model library is loaded with `dlopen`, and the functions are looked up by
//! name with `dlsym`, once, when the model opens. Holding plain function pointers
//! means a call is a direct jump, with no lookup per call. The table is only valid
//! while the library it came from stays loaded, so [`crate::model`] keeps the
//! library and the table together.

use std::ffi::CString;

use libloading::os::unix::Library as UnixLibrary;
use libloading::Library;

use crate::abi::*;
use crate::error::Error;

/// Every function the crate calls, as function pointers.
pub(crate) struct Api {
    pub(crate) create: CreateFn,
    pub(crate) delete_container: DeleteContainerFn,
    pub(crate) run: RunFn,
    pub(crate) num_outputs: NumOutputsFn,
    pub(crate) create_blob: CreateBlobFn,
    pub(crate) data_ptr: DataPtrFn,
    pub(crate) dim: DimFn,
    pub(crate) sizes: SizesFn,
    pub(crate) strides: StridesFn,
    pub(crate) dtype: DtypeFn,
    pub(crate) delete_tensor: DeleteTensorFn,
    pub(crate) dtype_float32: ConstI32Fn,
    pub(crate) device_cpu: ConstI32Fn,
}

impl Api {
    /// Looks every function up. `AOTInductor*` symbols come from the model library;
    /// the `aoti_torch_*` helpers are exported by libtorch itself, which is linked into
    /// the process, so they are looked up globally rather than in the model library.
    /// Fails with [`Error::MissingSymbol`] when one isn't there.
    pub(crate) fn load(lib: &Library) -> Result<Api, Error> {
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

/// Looks up `name` in the library [lib], as a function pointer of type `T`.
fn sym<T: Copy>(lib: &Library, name: &'static str) -> Result<T, Error> {
    let c_name = CString::new(name).expect("symbol names contain no NUL");
    let symbol = unsafe { lib.get::<T>(c_name.as_bytes_with_nul()) }.map_err(|e| Error::MissingSymbol {
        name,
        reason: e.to_string(),
    })?;
    Ok(*symbol)
}

/// Looks up `name` among the symbols of the whole process, where libtorch's helpers are.
fn global_sym<T: Copy>(lib: &UnixLibrary, name: &'static str) -> Result<T, Error> {
    let c_name = CString::new(name).expect("symbol names contain no NUL");
    let symbol = unsafe { lib.get::<T>(c_name.as_bytes_with_nul()) }.map_err(|e| Error::MissingSymbol {
        name,
        reason: e.to_string(),
    })?;
    Ok(*symbol)
}

/// Turns a libtorch status into a result, naming the call that failed.
pub(crate) fn check(call: &'static str, code: i32) -> Result<(), Error> {
    if code == SUCCESS {
        Ok(())
    } else {
        Err(Error::Aoti { call, code })
    }
}
