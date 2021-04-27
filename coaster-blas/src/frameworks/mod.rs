//! Provides the specific Framework implementations for the Library Operations.

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "native")]
pub mod native;
#[cfg(feature = "opencl")]
pub mod opencl;


#[cfg(not(feature="cuda"))]
use log as _;
