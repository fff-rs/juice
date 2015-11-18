//! Provides a simple and unified API to run fast and highly parallel computations on different
//! devices such as CPUs and GPUs and accross different computation languages such as OpenCL and
//! CUDA.
//!
//! Collenchyma was started at [Autumn][autumn] to create an easy and performant abstraction over
//! different backends for the Machine Intelligence Framework [Leaf][leaf], that has no hard
//! dependency on any driver or libraries and could easily be used without the need for a long and
//! painful build process.
//!
//! The naming of processes, concepts and most other parts is oriented towards the OpenCL 1.2
//! standard described in [this Glossary p.14][glossary]. But don't worry, the functionality of the
//! fundamentel modules, is described in this documentation as well.
//!
//! [autumn]: http://autumnai.com
//! [leaf]: https://github.com/autumnai/leaf
//! [glossary]: https://www.khronos.org/registry/cl/specs/opencl-1.2.pdf
#![feature(plugin)]
#![plugin(clippy)]
#![allow(dead_code)]
#![feature(associated_consts)]
#![feature(unboxed_closures)]
#![feature(static_mutex)]
#![deny(missing_docs,
        missing_debug_implementations, missing_copy_implementations,
        trivial_casts, trivial_numeric_casts,
        unused_import_braces, unused_qualifications)]

extern crate libc;
#[macro_use]
extern crate enum_primitive;
extern crate num;

pub mod backend;
pub mod buffer;
pub mod device;
pub mod framework;
pub mod frameworks;
pub mod error;
pub mod program;
pub mod programs;
pub mod kernel;
