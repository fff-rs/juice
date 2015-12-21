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
//! trait, that defines all the available methods, which will later be available through the Backend, as
//! the Plugin Trait is implemented for the Collenchyma Backend. The operations take as arguments one or many
//! SharedTensors, holding the data over which the operation should happen, and none or one Operation Configuration.
//!
//! This Plugin trait is then implemented for the Backend, and specificially for each Computation Language, such as
//! CUDA, OpenCL or common host CPU.
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
//! This again is just plain Rust - nothing fancy about it. Example:
//!
//! ```rust
//! extern crate collenchyma as co;
//! extern crate collenchyma_nn as nn;
//! use co::backend::{Backend, BackendConfig};
//! use co::framework::IFramework;
//! use co::frameworks::Cuda;
//! use co::tensor::SharedTensor;
//! use nn::*;
//! fn main() {
//!     // Initialize a CUDA Backend.
//!     // Usually you would not use CUDA but let it pick what is available on the machine.
//!     let framework = Cuda::new();
//!     let hardwares = framework.hardwares();
//!     let backend_config = BackendConfig::new(framework, hardwares);
//!     let backend = Backend::new(backend_config).unwrap();
//!     // Initialize two SharedTensors.
//!     // Usually you would want also fill them with data.
//!     let mut x = SharedTensor::<f32>::new(backend.device(), &(1, 1, 3)).unwrap();
//!     let mut result = SharedTensor::<f32>::new(backend.device(), &(1, 1, 3)).unwrap();
//!     // Use the operation provided by this Plugin.
//!     backend.sigmoid(&mut x, &mut result);
//! }
//! ```
//!
//! ## Provided Operations
//!
//! This Plugins provides the following operations. (Forward + Backward)
//! A `-` means not yet implemented.
//!
//! | Operation            | CUDA       | OpenCL    | Native    |
//! |---	               |---	        |---        |---        |
//! | Sigmoid  	           | cuDNN v3  	| -  	    | -  	   	|
//! | ReLU  	           | cuDNN v3   | -  	    | - 	    |
//! | Tanh  	   	       | cudNN v3   | - 	    | -         |
//! |   	   	           |  	        |  	        |           |
//! | Normalization (LRN)  | cudNN v3   | - 	    | -         |
//! |   	   	           |  	        |  	        |           |
//! | Convolution          | cudNN v3   | - 	    | -         |
//! |   	   	           |  	        |  	        |           |
//! | Softmax              | cudNN v3   | - 	    | -         |
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

pub use plugin::{NN, NNOperationConfig};

mod plugin;
pub mod frameworks;
