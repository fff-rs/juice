//! Provides a [Collenchyma][collenchyma] Plugin, to extend Collenchyma with Neural Network related
//! operations such as convolutions, pooling, ReLU, etc. A full list of operations provided by this Plugin,
//! can be found at the [provided Operations section](#operations).
//!
//! ## Overview
//!
//! This Collenchyma Plugin extends Collenchyma's Backend with NN related methods/operations. This allows
//! you to run, these operations (and therefore your application) on your local machine as well as on servers,
//! mobiles or any other machine (as if they were written for common CPU execution), while
//! receiving the significant performance increases (usually one-to-two orders of magnitutde), by
//! executing the operations on special purpose hardware such as GPUs - if they are available. Usage examples
//! can be found in the next section.
//!
//! The architecture of a Plugin is quite easy. It defines one Plugin Trait, in this case the `NN`
//! trait, which implements basic functionality for initialization and multiple Plugin Operation Traits which define the
//! methods which are going to be available on the Backed, as the Plugin Trait as well as the Plugin Operations Traits
//! are implemented for the Collenchyma Backends (CUDA, OpenCL, Native). The operations take as arguments one or many
//! SharedTensors, holding the data over which the operation should happen, and none or one Operation Configuration.
//!
//! ## Usage
//!
//! Using this Collenchyma Plugin is like using any other Rust crate - super easy. In your `Cargo.toml` define dependencies
//! to both [Collenchyma][collenchyma] (if not yet happend) and the plugin. For example:
//!
//! ```toml
//! [dependencies]
//! collenchyma: "latest",
//! collenchyma_nn: "latest"
//! ```
//!
//! The next and final step is, bringing the crates and the important parts into the scope of your application module.
//! This again is just plain Rust - nothing fancy about it. Now a complete example:
//!
//! ```rust
//! # #![allow(dead_code)]
//! extern crate collenchyma as co;
//! extern crate collenchyma_nn as nn;
//! # #[cfg(feature = "cuda")]
//! # mod cuda {
//! use co::backend::{Backend, BackendConfig};
//! use co::framework::IFramework;
//! use co::frameworks::{Cuda, Native};
//! use co::memory::MemoryType;
//! use co::tensor::SharedTensor;
//! use nn::*;
//!
//! fn write_to_memory<T: Copy>(mem: &mut MemoryType, data: &[T]) {
//!     if let &mut MemoryType::Native(ref mut mem) = mem {
//!         let mut mem_buffer = mem.as_mut_slice::<T>();
//!         for (index, datum) in data.iter().enumerate() {
//!             mem_buffer[index] = *datum;
//!         }
//!     }
//! }
//!
//! pub fn main() {
//!     // Initialize a CUDA Backend.
//!     // Usually you would not use CUDA but let Collenchyma pick what is available on the machine.
//!     let framework = Cuda::new();
//!     let hardwares = framework.hardwares();
//!     let backend_config = BackendConfig::new(framework, hardwares);
//!     let backend = Backend::new(backend_config).unwrap();
//!     // Initialize two SharedTensors.
//!     let mut x = SharedTensor::<f32>::new(backend.device(), &(1, 1, 3)).unwrap();
//!     let mut result = SharedTensor::<f32>::new(backend.device(), &(1, 1, 3)).unwrap();
//!     // Fill `x` with some data.
//!     let payload: &[f32] = &::std::iter::repeat(1f32).take(x.capacity()).collect::<Vec<f32>>();
//!     let native = Native::new();
//!     let cpu = native.new_device(native.hardwares()).unwrap();
//!     x.add_device(&cpu).unwrap(); // Add native host memory
//!     x.sync(&cpu).unwrap(); // Sync to native host memory
//!     write_to_memory(x.get_mut(&cpu).unwrap(), payload); // Write to native host memory.
//!     x.sync(backend.device()).unwrap(); // Sync the data to the CUDA device.
//!     // Run the sigmoid operation, provided by the NN Plugin, on your CUDA enabled GPU.
//!     backend.sigmoid(&mut x, &mut result).unwrap();
//!     // See the result.
//!     result.add_device(&cpu).unwrap(); // Add native host memory
//!     result.sync(&cpu).unwrap(); // Sync the result to host memory.
//!     println!("{:?}", result.get(&cpu).unwrap().as_native().unwrap().as_slice::<f64>());
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
//! This Plugins provides the following operations. (Forward + Backward)
//! A `-` means not yet implemented.
//!
//! | Operation            | CUDA       | OpenCL    | Native    |
//! |---	               |---	        |---        |---        |
//! | Sigmoid  	           | cuDNN v3  	| -  	    | Rust  	|
//! | SigmoidPointwise     | cuDNN v3  	| -  	    |   	    |
//! | ReLU  	           | cuDNN v3   | -  	    | Rust 	    |
//! | ReLUPointwise        | cuDNN v3  	| -  	    |   	    |
//! | Tanh  	   	       | cudNN v3   | - 	    | Rust      |
//! | TanhPointwise        | cuDNN v3  	| -  	    |   	    |
//! |   	   	           |  	        |  	        |           |
//! | Normalization (LRN)  | cudNN v3   | - 	    | -         |
//! |   	   	           |  	        |  	        |           |
//! | Convolution          | cudNN v3   | - 	    | -         |
//! |   	   	           |  	        |  	        |           |
//! | Softmax              | cudNN v3   | - 	    | Rust      |
//! | LogSoftmax           | cudNN v3   | - 	    | Rust      |
//! |   	   	           |  	        |  	        |           |
//! | Pooling Max          | cudNN v3   | - 	    | -         |
//! | Pooling Avg          | cudNN v3   | - 	    | -         |
//!
//! [collenchyma]: https://github.com/autumnai/collenchyma
//! [collenchyma-docs]: http://autumnai.github.io/collenchyma
//! [blas-source]: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
#![cfg_attr(lint, feature(plugin))]
#![cfg_attr(lint, plugin(clippy))]
#![allow(dead_code)]
#![deny(missing_docs,
        missing_debug_implementations, missing_copy_implementations,
        trivial_casts, trivial_numeric_casts,
        unused_import_braces, unused_qualifications)]

extern crate collenchyma as co;
#[cfg(feature = "cuda")]
extern crate cudnn;
extern crate libc;
#[macro_use]
extern crate lazy_static;

pub use plugin::*;

mod plugin;
pub mod frameworks;
