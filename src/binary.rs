//! Provides the generic functionality for a collection of backend-agnostic operations.
//!
//! A binary defines one or usually many operations, sharing related functionalities.
//! Two examples would be [BLAS][blas] (Basic Linear Algebra Subprograms) or [cuDNN][cudnn],
//! providing operations for specific applications.
//! The difference between these programs and programs defined in Collenchyma is, that a program in
//! Collenchyma is backend-agnostic. Whereas e.g. cuDNN only runs on the CUDA Framework and
//! therefore only CUDA-supported devices, a DNN program in Collenchyma (sharing the same
//! operations as cuDNN) would run on other backends and [frameworks][frameworks] such as Host CPU,
//! OpenCL as well.
//!
//! A program is either build from text or binary or can use already build (and cached) kernel
//! functions, removing the need for various system libraries, which often results in a tedious and
//! frustrating setup experience.
//!
//! You are ususally not interacting with a program itself, but rather use it to construct the
//! needed backend-agnostic operations aka. kernels, which can then be run and parallelized via an
//! unified interface - [backend.call()][backend-call].
//!
//! ## Architecture
//!
//! In order to be extendable and provide a truly consistent and unified API to run operations on
//! any backend, a program is closely related to a [kernel][kernel]. A program is first a container
//! for multiple kernel functions providing access to them and secondly an abstraction for building
//! the kernel functions on the underlying device(s), so that they can be called.
//!
//! A program makes these kernel functions, which may be
//!
//! * written in Rust (e.g. for Host CPU execution),
//! * accessible through Rust (e.g. ffi for Host CPU execution) or
//! * writen in other languages such as OpenCL or CUDA, etc.
//!
//! controllable in Rust trough a unified Interface. A program is therefore one of the most
//! critical aspects of Collenchyma's architecture.
//!
//! The specific programs (BLAS, DNN, etc.) implement the shared functionality for all
//! or some of the supported [frameworks][frameworks]. This is done as the program provides traits
//! which, then can be implemented for each specific framework. This generates a nice, native
//! Rust access to these kernel methods, with a unified interface for every backend and completely
//! controllable at runtime.
//!
//! ## Development
//!
//! The here provided funcionality is used to construct specific Collenchyma programs, which gets
//! used to construct the basic programs that come shipped with Collenchyma, but also allows you to
//! define and run your own backend-agnostic programs as well.
//!
//! [blas]: http://www.netlib.org/blas/
//! [cudnn]: https://developer.nvidia.com/cudnn
//! [frameworks]: ../frameworks/index.html
//! [backend-call]: ../backend/struct.Backend.html#method.call
//! [kernel]: ../kernel/index.html

use operation::IOperation;
use library::ILibrary;
use std::collections::HashMap;

/// Defines the functionality for turning a library into backend-specific, executable operations.
pub trait IBinary {
    /// The Operation representation for this Binary.
    type O: IOperation;
    /// The Library representation for this Binary.
    type L: ILibrary;
    /// Returns the unique identifier of the Binary.
    fn id(&self) -> isize;
    /// Creates a HashMap of available, ready-to-use operations, based on the provided library and
    /// tailored for a framework.
    fn create_operations() -> HashMap<String, Self::O>;
}
