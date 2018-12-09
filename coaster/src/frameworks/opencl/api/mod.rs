//! Provides a safe wrapper around OpenCL.

pub use self::error::Error;

#[derive(Debug, Copy, Clone)]
/// Defines the OpenCL API.
pub struct API;

mod error;
mod context;
mod device;
mod memory;
mod platform;
mod queue;
mod ffi;
pub mod types;
