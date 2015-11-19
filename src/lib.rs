//! Provides a simple and unified API to run fast and highly parallel computations on different
//! devices such as CPUs and GPUs, accross different computation languages such as OpenCL and
//! CUDA and allows you to swap your backend on run-time.
//!
//! Collenchyma was started at [Autumn][autumn] to create an easy and performant abstraction over
//! different backends for the Machine Intelligence Framework [Leaf][leaf], with no hard
//! dependency on any driver or libraries so that it can easily be used without the need for a
//! long and painful build process.
//!
//! ## Abstract
//!
//! Code often is executed on the native CPU, but could be executed on other devices such as GPUs
//! and Accelerators as well. These devices are accessable through frameworks like OpenCL and CUDA
//! but have a more complicated interfaces than your every-day native CPU
//! which makes the use of these devices a painful experience. Some of the pain points, when
//! writing such device code, are:
//!
//! * non-portable: frameworks have different interfaces, devices support different versions and
//! machines might have different hardware - all this leads to code that will be executable only on
//! a very specific set of machines and platforms.
//! * steep learning curve: executing code on a device through a framework is quite different to
//! running code on the native CPU and comes with a lot of hurdles. OpenCLs 1.2 specification for
//! example has close to 400 pages.
//! * custom code: integrating support for devices into your project, requires the need for writing
//! a lot of custom code e.g. kernels, memory management, genereal business logic.
//!
//! But writing code for devices would often be a good choice as these devices can execute many
//! operations a lot faster than the native CPU. GPUs for example can execute operations roughly
//! one to two orders of magnitudes faster, thanks to better support of parallising operations.
//! OpenCL and CUDA make parallising operations super easy.
//!
//! With Collenchyma we eleminate the pain points of writing device code, so you can run your code
//! like any other Rust code, don't need to learn about kernels, events, queues or memory
//! synchronization, and can deploy your code with ease to servers, desktops or mobiles and
//! your code will make full use of the underlying hardware.
//!
//! ## Development
//!
//! At the moment Collenchyma itself will provide Rust APIs for the important frameworks - OpenCL
//! and CUDA. One step we are looking out for is to seperate OpenCL and CUDA into their own crate.
//! Something similar to [Glium][glium].
//!
//! [autumn]: http://autumnai.com
//! [leaf]: https://github.com/autumnai/leaf
//! [glium]: https://github.com/tomaka/glium
#![feature(plugin)]
#![plugin(clippy)]
#![allow(dead_code)]
#![feature(associated_consts)]
#![feature(associated_type_defaults)]
#![feature(unboxed_closures)]
#![feature(static_mutex)]
#![deny(missing_docs,
        missing_debug_implementations, missing_copy_implementations,
        trivial_casts, trivial_numeric_casts,
        unused_import_braces, unused_qualifications)]

extern crate libc;
#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate enum_primitive;
extern crate num;
extern crate byteorder;
extern crate rblas as blas;

pub mod backend;
pub mod device;
pub mod hardware;
pub mod framework;
pub mod frameworks;
pub mod memory;
pub mod library;
pub mod libraries;
pub mod shared_memory;
pub mod operation;
pub mod binary;
