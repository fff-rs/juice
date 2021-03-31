//! Provides a safe wrapper around OpenCL.

pub use self::error::Error;

#[derive(Debug, Copy, Clone)]
/// Defines the OpenCL API.
pub struct API;

mod context;
mod device;
mod error;
mod ffi;
mod memory;
mod platform;
mod queue;
pub mod types;
