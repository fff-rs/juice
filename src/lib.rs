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
//! Operations get introduced by a [Library][library]. A Library provides which operations are available and provides
//! the finished implementation of an Operation. This allows us, to implement the Library directly on the Backend, which
//! gives us a native Rust interface to execute our Operations. The interface is like any other interface e.g. to
//! execute the dot product operation we can simply call `backend.dot(...)`. If the dot Operation is executed on e.g.
//! one or many GPUs or CPUs depends solely on how you configured the Backend or if you did not further specify which
//! Framework and Hardware to use, solely on the machine you execute the dot Operation on. A Library exposes the
//! Operation traits, which define one `computation` method, that is further defined by the Framework, where the
//! exact Operation execution is stated. In the field of Operation there is one more component
//! - the [Binary][binary]. As - different to executing code on the native CPU - devices need to compile and build the
//! Operation, which is a significant part of a Framework, each Framework exposes a Binary representation, which mounts
//! the Operations exposed by a Library.
//!
//! The last peace of Collenchyma is the [Memory][memory]. A Operation happens over data, but this data needs to be
//! accessable by the device on which the Operation is executed. The process is therefore often, that memory space needs
//! to be allocated on the device and then in a later step, synced from the host to the device or from
//! the device back to the host. Thanks to [SharedMemory][shared-memory] we do not have to care about memory management
//! between devices for the execution of Operations. SharedMemory tracks and automatically manages data and it's memory
//! accross devices, which is often the host and the Device. But it can also be passed around to different Backends.
//! Operations take as arguments SharedMemory and handle the synchronization and allocation for you.
//!
//! ## Examples
//!
//! Backend with custom defined Framework and Device.
//!
//! ```
//! extern crate collenchyma as co;
//! use co::framework::IFramework;
//! use co::backend::{Backend, BackendConfig};
//! use co::frameworks::Native;
//! #[allow(unused_variables)]
//! fn main() {
//!     let framework = Native::new(); // Initialize the Framework
//!     let hardwares = framework.hardwares(); // Now you can obtain a list of available hardware for that Framework.
//!     // Create the custom Backend by providing a Framework and one or many Hardwares.
//!     let backend_config = BackendConfig::new(framework, hardwares);
//!     let backend = Backend::new(backend_config);
//!     // You can now execute all the operations available, e.g.
//!     // backend.dot(x, y);
//! }
//! ```
//!
//! Machine-agnostic Backend.
//!
//! ```
//! extern crate collenchyma as co;
//! fn main() {
//!     // Not yet implemented.
//!     // No need to provide a Backend Configuration.
//!     // let backend = Backend::new(None);
//!     // You can now execute all the operations available, e.g.
//!     // backend.dot(x, y);
//! }
//! ```
//!
//! ## Development
//!
//! At the moment Collenchyma itself will provide Rust APIs for the important frameworks - OpenCL
//! and CUDA. One step we are looking out for is to seperate OpenCL and CUDA into their own crate.
//! Something similar to [Glium][glium].
//!
//! Every operation exposed via a library and implemented on the backend, should take as the last argument an
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
//! [library]: ./libraries/index.html
//! [memory]: ./memory/index.html
//! [shared-memory]: ./shared-memory/index.html
#![cfg_attr(lint, feature(plugin))]
#![cfg_attr(lint, plugin(clippy))]
#![feature(link_args)]
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
extern crate rblas as blas;

#[macro_use]
pub mod libraries;
pub mod backend;
pub mod device;
pub mod hardware;
pub mod framework;
pub mod frameworks;
pub mod memory;
pub mod shared_memory;
pub mod operation;
pub mod binary;
pub mod error;
