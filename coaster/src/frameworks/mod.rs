//! Exposes the specific Framework implementations.

#[cfg(feature = "cuda")]
pub use self::cuda::Cuda;
pub use self::native::Native;
#[cfg(feature = "opencl")]
pub use self::opencl::OpenCL;

#[cfg(feature = "cuda")]
pub mod cuda;
pub mod native;
#[cfg(feature = "opencl")]
pub mod opencl;
