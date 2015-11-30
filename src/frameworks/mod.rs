//! Exposes the specific Framework implementations.

pub use self::native::Native;
pub use self::opencl::OpenCL;
pub use self::cuda::Cuda;

pub mod native;
pub mod opencl;
pub mod cuda;
