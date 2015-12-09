//! Provides backend-agnostic Neural Network operations for [Collenchyma][collenchyma].
//!
//! ## Overview
//!
//! A Collenchyma Plugin describes the functionality through three types of traits.
//!
//! * __PluginTrait__ -> INn<br/>
//! This trait provides 'provided methods', which already specify the exact, backend-agnostic
//! behavior of an Operation. These come in two forms `operation()` and `operation_plain()`,
//! where the first takes care of full memory management and the later one just provides the computation
//! without any memory management. In some scenarios you would like to use the plain operation for faster
//! exection.
//!
//! * __BinaryTrait__ -> IBlasBinary<br>
//! The binary trait provides the actual and potentially initialized Functions, which are able to compute
//! the Operations (as they implement the OperationTrait).
//!
//! * __OperationTrait__ -> e.g. IOperationSigmoid<br/>
//! The PluginTrait can provide 'provided methods', thanks to the OperationTrait. The OperationTrait,
//! has one required method `compute` which every Framework Function will implement on it's own way.
//!
//! Beside these traits a Collenchyma Plugin might also use macros for faster
//! implementation for various Collenchyma Frameworks such as CUDA, OpenCL or common host CPU.
//!
//! Beside these generic functionality through traits, a Plugin also extends the Collenchyma Backend
//! with implementations of the generic functionality for the Collenchyma Frameworks.
//!
//! For more information, give the [Collenchyma docs][collenchyma-docs] a visit.
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

extern crate collenchyma;
#[cfg(feature = "cuda")]
extern crate cudnn;
#[macro_use]
extern crate lazy_static;

pub mod library;
pub mod binary;
pub mod operation;
#[macro_use]
pub mod helper;
mod frameworks;
