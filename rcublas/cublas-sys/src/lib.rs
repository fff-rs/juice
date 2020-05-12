mod generated;

pub use crate::generated::*;

unsafe impl std::marker::Send for crate::generated::cublasContext {}


pub use ptr_origin_tracker as tracker;

pub use tracker::prelude::*;

tracker::setup!(cublasContext);