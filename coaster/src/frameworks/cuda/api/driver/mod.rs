//! Provides a safe wrapper around the CUDA Driver API.

pub use self::error::Error;

#[derive(Debug, Copy, Clone)]
/// Defines the Cuda API.
pub struct API;

mod error;
mod context;
mod device;
mod memory;
pub mod ffi;
mod utils;
