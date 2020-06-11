//! Provides backend-agnostic BLAS operations for [Coaster][coaster].
//!
//! BLAS (Basic Linear Algebra Subprograms) is a specification that prescribes a set of low-level
//! routines for performing common linear algebra operations such as vector addition, scalar
//! multiplication, dot products, linear combinations, and matrix multiplication. They are the de
//! facto standard low-level routines for linear algebra libraries; the routines have bindings for
//! both C and Fortran. Although the BLAS specification is general, BLAS implementations are often
//! optimized for speed on a particular machine, so using them can bring substantial performance
//! benefits. BLAS implementations will take advantage of special floating point hardware such as
//! vector registers or SIMD instructions.<br/>
//! [Source][blas-source]
//!
//! ## Overview
//!
//! A Coaster Plugin describes the functionality through three types of traits.
//!
//! * __PluginTrait__ -> IBlas<br/>
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
//! * __OperationTrait__ -> e.g. IOperationDot<br/>
//! The PluginTrait can provide 'provided methods', thanks to the OperationTrait. The OperationTrait,
//! has one required method `compute` which every Framework Function will implement on it's own way.
//!
//! Beside these traits a Coaster Plugin might also use macros for faster
//! implementation for various Coaster Frameworks such as CUDA, OpenCL or common host CPU.
//!
//! Beside these generic functionality through traits, a Plugin also extends the Coaster Backend
//! with implementations of the generic functionality for the Coaster Frameworks.
//!
//! For more information, give the [Coaster docs][coaster-docs] a visit.
//!
//! [coaster]: https://github.com/spearow/coaster
//! [coaster-docs]: https://spearow.github.io/coaster
//! [blas-source]: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
#![allow(dead_code)]
#![deny(
    clippy::missing_docs,
    clippy::missing_debug_implementations,
    clippy::missing_copy_implementations,
    clippy::trivial_casts,
    clippy::trivial_numeric_casts,
    clippy::unsafe_code,
    clippy::unused_import_braces,
    clippy::unused_qualifications,
    clippy::complexity
)]

extern crate coaster;
#[cfg(feature = "cuda")]
extern crate rcublas as cublas;
#[cfg(feature = "native")]
extern crate rust_blas as rblas;

pub mod binary;
pub mod frameworks;
pub mod operation;
pub mod plugin;
pub mod transpose;
