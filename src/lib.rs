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
//! operations a lot faster than the native CPUs. GPUs for example can execute operations roughly
//! one to two orders of magnitudes faster, thanks to better support of parallising operations.
//! OpenCL and CUDA make parallising operations super easy.
//!
//! With Collenchyma we eleminate the pain points of writing device code, so you can run your code
//! like any other Rust code, don't need to learn about kernels, events, or memory
//! synchronization, and can deploy your code with ease to servers, desktops or mobiles and
//! your code will make full use of the underlying hardware.
//!
//! ## Architecture
//!
//! The single entry point of Collenchyma is a [Backend][backend]. A Backend is agnostic over the [Device][device] it
//! runs [Operations][operation] on. In order to be agnostic over the Device, such as native host CPU, GPUs,
//! Accelerators or other types of [Hardware][hardware], the Backend needs to be agnostic over the
//! [Framework][framework] as well. A Framework is a computation language such as OpenCL, Cuda or the native programming
//! language. The Framework is important, as it provides us with the interface to turn Hardware into Devices and
//! therefore, among other things, execute Operations on the created Device. With a Framework, we get access to Hardware
//! as long as the Hardware supports the Framework. As different vendors of Hardware use different
//! Frameworks, it becomes important that the Backend is agnostic over the Framework, which allows us, that we can
//! really run computations on any machine such as servers, desktops and mobiles without the need to worry about what
//! Hardware is available on the machine. That gives us the freedom to write code once and deploy it on different
//! machines where it will execute on the most potent Hardware by default.
//!
//! Operations get introduced by a [Plugin][plugin]. A Plugin extends your Backend with ready-to-execute Operations.
//! All you need to do is, providing these Collenchyma Plugin crates alongside the Collenchyma crate in your Cargo
//! file. Your Backend will than be extend with the operations provided by the Plugin. The interface is just common
//! Rust e.g. to execute the dot product operation of the [Collenchyma-BLAS][collenchyma-blas] Plugin,
//! we can simply call `backend.dot(...)`. If the dot Operation is executed on e.g.
//! one or many GPUs or CPUs depends solely on how you configured the Backend or if you did not further specify which
//! Framework and Hardware to use, solely on the machine you execute the dot Operation on. In the field of Operations
//! is one more component - the [Binary][binary]. As - different to executing code on the native CPU - devices need
//! to compile and build the Operation manually at run-time, which makes a significant part of a Framework, we need
//! an initlizable instance for holding the state and compiled Operations, wich the Binary is good for.
//!
//! The last peace of Collenchyma is the [Memory][memory]. A Operation happens over data, but this data needs to be
//! accessable by the device on which the Operation is executed. The process is therefore often, that memory space needs
//! to be allocated on the device and then in a later step, synced from the host to the device or from
//! the device back to the host. Thanks to the [Tensor][tensor] we do not have to care about memory management
//! between devices for the execution of Operations. Tensor tracks and automatically manages data and it's memory
//! accross devices, which is often the host and the Device. But it can also be passed around to different Backends.
//! Operations take as arguments Tensors and handle the synchronization and allocation for you.
//!
//! ## Examples
//!
//! This example requires the Collenchyma NN Plugin, for Neural Network related operations, to work.
//!
//! ```ignore
//! extern crate collenchyma as co;
//! extern crate collenchyma_nn as nn;
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
//! fn main() {
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
//! ```
//!
//! ## Development
//!
//! At the moment Collenchyma itself will provide Rust APIs for the important frameworks - OpenCL
//! and CUDA. One step we are looking out for is to seperate OpenCL and CUDA into their own crate.
//! Something similar to [Glium][glium].
//!
//! Every operation exposed via a Plugin and implemented on the backend, should take as the last argument an
//! `Option<OperationConfig>` to specify custom parallelisation behaviour and tracking the operation via events.
//!
//! When initializing a new Backend from a BackendConfig you might not want to specify the Framework, which is currently
//! mandatory. Leaving it blank, the Backend would try to use the most potent Framework given the underlying hardware,
//! which would be probably in this order Cuda -> OpenCL -> Native. The setup might take longer, as every framework
//! needs to be checked, and devices be loaded in order to identify the best setup. But this would allow, that you
//! really could deploy a Collenchyma-backed application to almost any hardware - server, desktops, mobiles.
//!
//! [autumn]: http://autumnai.com
//! [leaf]: https://github.com/autumnai/leaf
//! [glium]: https://github.com/tomaka/glium
//! [backend]: ./backend/index.html
//! [device]: ./device/index.html
//! [binary]: ./binary/index.html
//! [operation]: ./operation/index.html
//! [hardware]: ./hardware/index.html
//! [framework]: ./framework/index.html
//! [plugin]: ./plugin/index.html
//! [collenchyma-blas]: https://github.com/autumnai/collenchyma-blas
//! [memory]: ./memory/index.html
//! [tensor]: ./tensor/index.html
#![cfg_attr(lint, feature(plugin))]
#![cfg_attr(lint, plugin(clippy))]
#![allow(dead_code)]
#![deny(missing_docs,
        missing_debug_implementations, missing_copy_implementations,
        trivial_casts, trivial_numeric_casts,
        unused_import_braces, unused_qualifications)]

extern crate libc;
#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate enum_primitive;
#[macro_use]
extern crate lazy_static;
extern crate num;
extern crate byteorder;
extern crate linear_map;

pub mod backend;
pub mod device;
pub mod hardware;
pub mod framework;
pub mod frameworks;
pub mod memory;
pub mod tensor;
pub mod operation;
pub mod binary;
pub mod error;
pub mod plugin;
