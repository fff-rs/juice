//! Provides a safe wrapper around the CUDA Driver API.

pub use self::error::Error;

#[derive(Debug, Copy, Clone)]
/// Defines the Cuda API.
pub struct API;

mod context;
mod device;
mod error;
pub mod ffi;
mod memory;
mod utils;
