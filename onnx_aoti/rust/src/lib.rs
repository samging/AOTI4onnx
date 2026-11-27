//! Loader for AOTInductor `.pt2` packages, for calling compiled models from
//! other languages.
//!
//! # Why a native layer exists
//!
//! amie runs on the JVM. Its ONNX models run on ONNX Runtime, which also has a
//! Java API. The AOTInductor runtime is different. A `.pt2` written by
//! `torch._inductor.aoti_compile_and_package` is a zip archive holding a compiled
//! `*.wrapper.so`: machine code that calls into libtorch's C++ runtime. The JVM
//! can't load that library or call its functions by itself, so a native layer has
//! to sit between the two. This crate is that layer. The Kotlin side reaches it
//! through the C ABI in [`ffi`], using Java's foreign function API
//! (`java.lang.foreign`).
//!
//! Rust is the language of this layer, not a requirement of the approach. The same
//! boundary could be written in C++, since libtorch is C++. Rust gives a
//! C-compatible export surface, and it keeps the unsafe calls into libtorch in a
//! few small functions that check their arguments first.
//!
//! # What a call does
//!
//! 1. [`AotiModel::load`] reads the `.pt2` as a zip archive, copies its
//!    `*.wrapper.so` to a temporary file, and `dlopen`s it. The copy is deleted
//!    as soon as it is loaded.
//! 2. The library exports `AOTInductorModelContainer*` functions. Creating a
//!    container gives `num_models` independent instances of the model.
//! 3. [`AotiModel::run`], [`AotiModel::run_into`] and [`AotiModel::run_view`] wrap
//!    the caller's float32 input as a libtorch tensor without copying it, run the
//!    container, and read the outputs back.
//! 4. Dropping the model deletes the container. The library is unloaded when no
//!    output view still holds it.
//!
//! The `aoti_torch_*` helpers are exported by libtorch itself, so they are looked
//! up in the process, not in the model library. libtorch must be linked into the
//! final binary, which `build.rs` does from the `LIBTORCH` directory.
//!
//! # Memory and threads
//!
//! Inputs are borrowed, never copied. Outputs must be float32 and contiguous. Each
//! [`AotiModel`] owns its container. Several threads can call one model at once,
//! up to its `num_models` instances.
//!
//! # Layout
//!
//! | Module | What it holds |
//! |---|---|
//! | `error` | The [`Error`] type every call returns |
//! | `abi` | The C signatures and constants of the functions called |
//! | `api` | The table of function pointers, and how each one is found |
//! | `shape` | Element counts and row-major strides |
//! | `tensor` | The copied [`Tensor`] and the checks that read a libtorch output |
//! | `model` | [`AotiModel`]: opening, running and freeing a model |
//! | `output` | [`OutputView`], an output left in libtorch's memory |
//! | [`ffi`] | The C ABI the Kotlin side calls |

pub mod ffi;

mod abi;
mod api;
mod error;
mod model;
mod output;
mod shape;
mod tensor;

pub use error::Error;
pub use model::AotiModel;
pub use output::OutputView;
pub use tensor::Tensor;
