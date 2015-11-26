//! Provides the generic functionality for backend-agnostic operations.
//!
//! A Kernel defines a function, a operation, that can be executed on various backends and
//! [frameworks][frameworks]. A Kernel can also easily be executed in parallel on multi-core
//! devices. A Kernel is a very similar to a usual function and has usually one or many attributes
//! over which the computation happens.
//!
//! You are usually not calling a kernel directly. To execute a kernel you would use the
//! [backend.call()][backend-call] method. Also you usually will not initialize your
//! kernels directly, this happens usually automatically at the initialization of a
//! [program][program].
//!
//! ## Architecture
//!
//! The Kernel is thin struct over the operation. Dependent on the framework, the kernel field will
//! either be a `KernelKind::Fn` and contains a real function that can be called directly from
//! Rust, which will be used for standard Host CPU framework or a `KernelKind::Obj` containing the
//! numeric KernelObject (id) through which it is acessable via the framework such as OpenCL, CUDA.
//!
//! After the kernel was initialized, usually at the initialization step of the program and
//! therefore at the initialization of the backend, it is available through the program, which
//! contains the hash of available kernels.
//!
//! Executing the kernel is the job of [backend.call()][backend-call], which awaits the index of
//! the program's kernel hash, to retrieve the kernel and then takes care of executing the kernel,
//! dependent on the KernelKind and framework.
//!
//! ## Development
//!
//! The here provided functionality is used to construct specific Collenchyma kernels, which are
//! sharing a similar logical functionality but can be executed across different backends. The
//! functionality provided by this module get used to construct the basic kernels that come shipped
//! with Collenchyma, but also allows you to define and run your own backend-agnostic kernels as
//! well.
//!
//! [frameworks]: ../frameworks/index.html
//! [backend-call]: ../backend/struct.Backend.html#method.call
//! [program]: ../program/index.html

/// Defines the functionality of an operation.
pub trait IOperation {}
