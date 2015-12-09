//! Exposes the specific Framework implementations.

#[cfg(feature = "native")]
pub use self::native::Native;
#[cfg(feature = "opencl")]
pub use self::opencl::OpenCL;
#[cfg(feature = "cuda")]
pub use self::cuda::Cuda;

#[cfg(feature = "native")]
pub mod native;
#[cfg(feature = "opencl")]
pub mod opencl;
#[cfg(feature = "cuda")]
pub mod cuda;
