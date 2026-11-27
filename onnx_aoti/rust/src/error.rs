//! The error type every fallible call returns.
//!
//! Errors come from four places: the filesystem and the `.pt2` zip (`Io`, `Zip`,
//! `NoWrapperInPackage`), the dynamic loader (`LoadLibrary`, `MissingSymbol`),
//! libtorch itself (`Aoti`, which carries the name of the failed call and its
//! status code), and the caller's arguments (the shape, dtype and size checks).

use std::fmt;
use std::io;

#[derive(Debug)]
pub enum Error {
    /// The package path can't be passed to the loader as UTF-8.
    InvalidPath,
    /// Reading the package or writing its extracted library failed.
    Io(io::Error),
    /// The `.pt2` file isn't a readable zip archive.
    Zip(zip::result::ZipError),
    /// The archive has no `*.wrapper.so`, so it wasn't made by AOTInductor.
    NoWrapperInPackage,
    /// `dlopen` refused the extracted library.
    LoadLibrary(libloading::Error),
    /// A symbol the runtime needs isn't exported, so the library is too old or not an AOTInductor one.
    MissingSymbol { name: &'static str, reason: String },
    /// A libtorch call returned a non-zero status.
    Aoti { call: &'static str, code: i32 },
    /// The shape given for the input is empty or has a non-positive dimension.
    InvalidShape(String),
    /// The input slice doesn't hold as many elements as its shape says.
    ShapeMismatch { expected: usize, got: usize },
    /// `run_into` was given a buffer of the wrong size for the output.
    OutputSizeMismatch { expected: usize, got: usize },
    /// `run_into` and `run_view` need a model with exactly one output.
    OutputCount(usize),
    /// The output isn't float32; the code is libtorch's dtype number.
    UnsupportedDtype(i32),
    /// The output's strides don't describe a row-major layout.
    NonContiguous,
    /// A model needs at least one instance to run.
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
