//! Provides safe API calls to CUDA's cuDNN library.
//!
//! Usually you will not use those calls directly, but acess them through,
//! the higher-level structs, exposed at the root of this crate, which provides
//! a more convenient and "rusty" interface.

pub mod activation;
pub mod convolution;
pub mod dropout;
pub mod lstm;
pub mod normalization;
pub mod pooling;
pub mod softmax;
pub mod tensor;
pub mod utils;
pub mod cuda;
