//! Provides the generic functionality for a collection of backend-agnostic operations.
//!
//! A program defines one or usually many kernel functions, sharing related functionalities.
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
//! You are ususally not interaction with a program itself, but rather use it to construct the
//! needed backend-agnostic operations aka. kernels, which can then be run and parallelized via
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
//! controllable in Rust via a unified Interface which is truly backend-agnostic. A program is
//! therefore one of the most critical aspects of Collenchyma's architecture.
//!
//! The specific programs (BLAS, DNN, etc.) implement the shared functionality for all
//! or some of the supported [frameworks][frameworks]. In order to know, which kernels are
//! accessible a specific program implements a build tree for it's kernels, which initializes
//! backend-agnostic kernels at the build step of a program - which is one of the first things
//! which will happen at the initialization of a new backend.
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

/// Defines the functionality that a Program implementation needs to support
pub trait Program {}
