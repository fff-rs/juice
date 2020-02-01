//! Provides a safe and convenient wrapper for the [CUDA cuDNN][cudnn] API.
//!
//! This crate (1.0.0) was developed against cuDNN v3.
//!
//! ## Architecture
//!
//! This crate provides three levels of entrace.
//!
//! **FFI**<br>
//! The `ffi` module exposes the foreign function interface and cuDNN specific types. Usually,
//! there should be no use to touch it if you only want to use cuDNN in you application. The ffi
//! is provided by the `rust-cudnn-sys` crate and gets reexported here.
//!
//! **Low-Level**<br>
//! The `api` module exposes already a complete and safe wrapper for the cuDNN API, including proper
//! Rust Errors. Usually there should be not need to use the `API` directly though, as the `Cudnn` module,
//! as described in the next block, provides all the API functionality but provides a more convenient interface.
//!
//! **High-Level**<br>
//! The `cudnn` module exposes the `Cudnn` struct, which provides a very convenient, easy-to-understand interface
//! for the cuDNN API. There should be not much need to obtain and read the cuDNN manual. Initialize the Cudnn
//! struct and you can call the available methods wich are representing all the available cuDNN operations.
//!
//! ## Examples
//!
//! ```
//! extern crate cudnn;
//! extern crate libc;
//! use cudnn::{Cudnn, TensorDescriptor};
//! use cudnn::utils::{ScalParams, DataType};
//! fn main() {
//!     // Initialize a new cuDNN context and allocates resources.
//!     let cudnn = Cudnn::new().unwrap();
//!     // Create a cuDNN Tensor Descriptor for `src` and `dest` memory.
//!     let src_desc = TensorDescriptor::new(&[2, 2, 2], &[4, 2, 1], DataType::Float).unwrap();
//!     let dest_desc = TensorDescriptor::new(&[2, 2, 2], &[4, 2, 1], DataType::Float).unwrap();
//!     // Obtain the `src` and memory pointer on the GPU.
//!     // NOTE: You wouldn't do it like that. You need to really allocate memory on the GPU with e.g. CUDA or Collenchyma.
//!     let src_data: *const ::libc::c_void = ::std::ptr::null();
//!     let dest_data: *mut ::libc::c_void = ::std::ptr::null_mut();
//!     // Now you can compute the forward sigmoid activation on your GPU.
//!     cudnn.sigmoid_forward::<f32>(&src_desc, src_data, &dest_desc, dest_data, ScalParams::default());
//! }
//! ```
//!
//! ## Notes
//!
//! rust-cudnn was developed at [Autumn][autumn] for the Rust Machine Intelligence Framework [Leaf][leaf].
//!
//! rust-cudnn is part of the High-Performance Computation Framework [Collenchyma][collenchyma], for the
//! [Neural Network Plugin][nn]. For an easy, unified interface for NN operations, such as those provided by
//! cuDNN, you might check out [Collenchyma][collenchyma].
//!
//! [cudnn]: https://developer.nvidia.com/cudnn
//! [autumn]: http://autumnai.com
//! [leaf]: https://github.com/autumnai/leaf
//! [collenchyma]: https://github.com/autumnai/collenchyma
//! [nn]: https://github.com/autumnai/collenchyma-nn
#![cfg_attr(lint, feature(plugin))]
#![cfg_attr(lint, plugin(clippy))]
#![allow(dead_code)]
#![deny(missing_docs,
        missing_debug_implementations, missing_copy_implementations,
        trivial_casts, trivial_numeric_casts,
        unused_import_braces, unused_qualifications)]

extern crate libc;
extern crate cudnn_sys as ffi;
extern crate num;

pub use ffi::*;
pub use self::cudnn::Cudnn;
pub use self::error::Error;
pub use self::tensor_descriptor::TensorDescriptor;
pub use self::convolution_descriptor::ConvolutionDescriptor;
pub use self::filter_descriptor::FilterDescriptor;
pub use self::normalization_descriptor::NormalizationDescriptor;
pub use self::pooling_descriptor::PoolingDescriptor;

#[derive(Debug, Copy, Clone)]
/// Defines the Cuda cuDNN API.
pub struct API;

mod cudnn;
mod error;
pub mod utils;
mod tensor_descriptor;
mod filter_descriptor;
mod normalization_descriptor;
mod pooling_descriptor;
mod convolution_descriptor;
mod api;
