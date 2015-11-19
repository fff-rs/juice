//! Provides the explicit implementation for [operations][operation] on a [backend][backend].
//! [operation]: ../operation/index.html
//! [backend]: ../backend/index.html
//!
//! A library can be directly implemented for a backend. A library provides 'provided methods',
//! meaning, that no code needs to be writen at the implementation of the library on a backend.
//!
//! A library plays a significant role in Collenchyma and allows for the native interface for
//! executing operations, which is no-different than calling a 'normal' Rust method. The library
//! declares the functionality, manages the shared memory arguments (synchronizations and memory
//! creations) and provides run-time notifications for operations which were not provided
//! at the creation of the backend.
//!
//! Examples of libraries would be [BLAS][blas] or a special purpose Neural Network library like
//! [cuDNN][cudnn]. But unlike BLAS or cuDNN, a Collenchyma library is completely backend-agnostic
//! and can operate on any framework-supported hardware.
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
//! which, then can be implemented for each of specific framework. This generates a nice, native
//! Rust access to these kernel methods, with a unified interface for every backend and completely
//! controllable at runtime.
//!
//! ## Development
//!
//! The here provided funcionality is used to construct specific Collenchyma libraries, which are
//! in use for constructing the basic libraries that come shipped with Collenchyma. The here
//! provided trait can also be used to define and run your own libraries.
//!
//! [blas]: http://www.netlib.org/blas/
//! [cudnn]: https://developer.nvidia.com/cudnn
//! [frameworks]: ../frameworks/index.html
//! [backend-call]: ../backend/struct.Backend.html#method.call
//! [kernel]: ../kernel/index.html

/// Defines the library behaviour, which is required by a [binary][binary].
/// [binary]: ../binary/index.html
pub trait ILibrary {}
