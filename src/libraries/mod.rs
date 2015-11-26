//! Exposes the explicit implementations of [operations][operation] on a [backend][backend].
//! [operation]: ../operation/index.html
//! [backend]: ../backend/index.html
//!
//! A library can be directly implemented for a backend. A library provides 'provided methods',
//! meaning, that no code needs to be writen at the implementation of the library on a backend.
//!
//! A library plays a significant role in Collenchyma and allows for the native interface for
//! executing operations, which is no-different than calling a 'normal' Rust method. The library
//! declares the functionality and automatically manages the shared memory arguments
//! (synchronizations and memory creations).
//!
//! Examples of libraries would be [BLAS][blas] or a special purpose Neural Network library like
//! [cuDNN][cudnn]. But unlike BLAS or cuDNN, a Collenchyma library is completely backend-agnostic
//! and can operate on any framework-supported hardware.
//!
//! Collenchyma ships with the most basic operations, but you should be able to easily write your
//! own backend-agnostic operations, too.
//!
//! [program]: ../program/index.html

pub mod blas;
