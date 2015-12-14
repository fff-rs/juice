//! Provides a safe and convenient wrapper around the [CUDA cuDNN][cudnn] API.
//!
//! This crate was developed against cuDNN v3.
//!
//! ## Architecture
//!
//! The `api` folder contains all the low-level functionality. Usually there should be no
//! need to use these methods, which are implemented for the [API struct][api], directly.
//! All the functionality should be accessible through the high-level structs exposed through
//! the modules in the root `src` folder.
//!
//! The `api` folder is structured like the modules in the root folder and expose safe methods,
//! around the cuDNN API - including proper Rust error messages for the cuDNN status types.
//! The `ffi.rs` file of the `api` folder contains the foreign function interface of cuDNN.
//!
//! [cudnn]: https://developer.nvidia.com/cudnn
//! [api]: ./struct.API.html
#![cfg_attr(lint, feature(plugin))]
#![cfg_attr(lint, plugin(clippy))]
#![allow(dead_code)]
#![deny(missing_docs,
        missing_debug_implementations, missing_copy_implementations,
        trivial_casts, trivial_numeric_casts,
        unused_import_braces, unused_qualifications)]

extern crate libc;
extern crate cudnn_sys as ffi;

pub use ffi::*;
pub use self::cudnn::Cudnn;
pub use self::error::Error;
pub use self::tensor_descriptor::{TensorDescriptor, DataType};
pub use self::convolution_descriptor::ConvolutionDescriptor;
pub use self::filter_descriptor::FilterDescriptor;

#[derive(Debug, Copy, Clone)]
/// Defines the Cuda cuDNN API.
pub struct API;

mod cudnn;
mod error;
pub mod utils;
mod tensor_descriptor;
mod filter_descriptor;
mod convolution_descriptor;
mod api;
