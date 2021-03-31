#![allow(dead_code)]

pub(crate) use rcublas_sys as ffi;

#[cfg(test)]
use coaster as co;

pub use api::Context;
pub use error::Error;

#[derive(Debug, Copy, Clone)]
/// Defines the cuBLAS API.
pub struct API;

pub mod api;
pub mod error;

#[cfg(test)]
pub(crate) mod chore;
