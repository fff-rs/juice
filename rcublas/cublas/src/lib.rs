#![allow(dead_code)]

extern crate rcublas_sys as ffi;

#[cfg(test)]
extern crate coaster as co;

#[macro_use]
extern crate lazy_static;

extern crate spin;

#[macro_use]
extern crate ctor;

pub use api::Context;
pub use error::Error;

#[derive(Debug, Copy, Clone)]
/// Defines the cuBLAS API.
pub struct API;

pub mod api;
pub mod error;
