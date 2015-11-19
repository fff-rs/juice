//! Exposes the specific Framework implementations.
//!
//! Examples would be OpenCL or CUDA.

pub use self::native::Native;
pub use self::opencl::OpenCL;

pub mod native;
pub mod opencl;
