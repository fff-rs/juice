//! Provides a safe wrapper around the CUDA cuDNN API.
//!
//! ## Architecture
//!
//! The `ffi.rs` file provides the foreign function interface for the cuDNN API.
//! The other files provide a safe wrapper around these foreign functions.
//! This includes clear Error types and a common interface of higher order, Rust structs.

pub use self::error::Error;

#[derive(Debug, Copy, Clone)]
/// Defines the Cuda cuDNN API.
pub struct API;

mod error;
pub mod ffi;
