//! Exposes the specific Framework implementations.
//!
//! Examples would be OpenCL or CUDA.

pub use self::host::Host;
pub use self::opencl::OpenCL;

pub mod host;
pub mod opencl;
pub mod opencl_ffi;
