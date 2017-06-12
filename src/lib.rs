//! Provides a [Coaster][coaster] Plugin, to extend Coaster with Neural Network related
//! operations such as convolutions, pooling, ReLU, etc. A full list of operations provided by this Plugin,
//! can be found at the [provided Operations section](#operations).
//!
//! ## Overview
//!
//! This Coaster Plugin extends Coaster's Backend with NN related methods/operations. This allows
//! you to run, these operations (and therefore your application) on your local machine as well as on servers,
//! mobiles or any other machine (as if they were written for common CPU execution), while
//! receiving the significant performance increases (usually one-to-two orders of magnitutde), by
//! executing the operations on special purpose hardware such as GPUs - if they are available. Usage examples
//! can be found in the next section.
//!
//! The architecture of a Plugin is quite easy. It defines one Plugin Trait, in this case the `NN`
//! trait, which implements basic functionality for initialization and multiple Plugin Operation Traits which define the
//! methods which are going to be available on the Backed, as the Plugin Trait as well as the Plugin Operations Traits
//! are implemented for the Coaster Backends (CUDA, OpenCL, Native). The operations take as arguments one or many
//! SharedTensors, holding the data over which the operation should happen, and none or one Operation Configuration.
//!
//! ## Usage
//!
//! An example on how to write some data into a SharedTensor and compute the result of the
//! sigmoid function for each value:
//!
//! ```rust
//! # #![allow(dead_code)]
//! extern crate coaster as co;
//! extern crate coaster_nn as nn;
//! # #[cfg(feature = "cuda")]
//! # mod cuda {
//! use co::prelude::*;
//! use co::frameworks::native::flatbox::FlatBox;
//! use nn::*;
//!
//! fn write_to_memory<T: Copy>(mem: &mut FlatBox, data: &[T]) {
//!     let mut mem_buffer = mem.as_mut_slice::<T>();
//!     for (index, datum) in data.iter().enumerate() {
//!         mem_buffer[index] = *datum;
//!     }
//! }
//!
//! pub fn main() {
//!     // Initialize a CUDA Backend.
//!     // Usually you would not use CUDA but let Coaster pick what is available on the machine.
//!     let backend = Backend::<Cuda>::default().unwrap();
//!     // Initialize two SharedTensors.
//!     let mut x = SharedTensor::<f32>::new(&(1, 1, 3));
//!     let mut result = SharedTensor::<f32>::new(&(1, 1, 3));
//!     // Fill `x` with some data.
//!     let payload: &[f32] = &::std::iter::repeat(1f32).take(x.capacity()).collect::<Vec<f32>>();
//!     let native = Native::new();
//!     let cpu = native.new_device(native.hardwares()).unwrap();
//!     write_to_memory(x.write_only(&cpu).unwrap(), payload); // Write to native host memory.
//!     // Run the sigmoid operation, provided by the NN Plugin, on your CUDA enabled GPU.
//!     backend.sigmoid(&mut x, &mut result).unwrap();
//!     // See the result.
//!     println!("{:?}", result.read(&cpu).unwrap().as_slice::<f64>());
//! }
//! # }
//! # #[cfg(not(feature = "cuda"))]
//! # mod cuda {
//! # pub fn main() {}
//! # }
//! #
//! # fn main() {
//! #     if cfg!(feature = "cuda") {
//! #         ::cuda::main();
//! #    }
//! # }
//! ```
//!
//! ## Provided Operations
//!
//! This Plugins provides the following operations. If not denoted otherwise, this means forward and backward
//! A `-` means not yet implemented.
//!

//! | Operation            | CUDA              | OpenCL    | Native    |
//! |---                   |---                |---        |---        |
//! | Sigmoid              | cuDNN v5 or later | -         | Rust      |
//! | SigmoidPointwise     | cuDNN v5 or later | -         | Rust      |
//! | ReLU                 | cuDNN v5 or later | -         | Rust      |
//! | ReLUPointwise        | cuDNN v5 or later | -         | Rust      |
//! | Tanh                 | cuDNN v5 or later | -         | Rust      |
//! | TanhPointwise        | cuDNN v5 or later | -         | Rust      |
//! |                      |                   |           |           |
//! | Normalization (LRN)  | cuDNN v5 or later | -         | -         |
//! |                      |                   |           |           |
//! | Convolution          | cuDNN v5 or later | -         | Rust(fwd) |
//! |                      |                   |           |           |
//! | Softmax              | cuDNN v5 or later | -         | Rust      |
//! | LogSoftmax           | cuDNN v5 or later | -         | Rust      |
//! |                      |                   |           |           |
//! | Pooling Max          | cuDNN v5 or later | -         | Rust(fwd) |
//! | Pooling Avg          | cuDNN v5 or later | -         | -         |
//!
//! [coaster]: https://github.com/ratpoison-io/coaster
//! [coaster-docs]: https://ratpoison.io/projects/coaster/documentation
//! [blas-source]: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
#![cfg_attr(feature = "unstable", feature(test))]
#![cfg_attr(lint, feature(plugin))]
#![cfg_attr(lint, plugin(clippy))]
#![allow(dead_code)]
#![deny(missing_docs,
        missing_debug_implementations, missing_copy_implementations,
        trivial_casts, trivial_numeric_casts,
        unused_import_braces, unused_qualifications)]

extern crate coaster as co;
#[cfg(feature = "cuda")]
extern crate cudnn;
extern crate libc;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;

#[cfg(test)]
extern crate rand;

pub use plugin::*;

mod plugin;
pub mod frameworks;

#[cfg(test)]
mod tests;
