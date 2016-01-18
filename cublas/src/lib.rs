#![allow(dead_code)]

extern crate cublas_sys as ffi;

#[cfg(test)]
extern crate collenchyma as co;

pub use error::Error;

#[derive(Debug, Copy, Clone)]
/// Defines the cuBLAS API.
pub struct API;

mod api;
mod error;
