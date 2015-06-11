// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.
#![feature(concat_idents)]
#![feature(core)]
#![feature(libc)]

//! BLAS bindings and wrappers.
//!
//! Bindings are split by BLAS level and contained in a module named `ll`
//! (stands for low level, not sure if that's the best name, but that's
//! what it is).
//!
//! Wrappers are split likewise. They are named after the function they wrap,
//! but capitalized and contained in their respective `ops` modules. To use
//! these wrappers, the appropriate traits must be implemented for the type.
//! These are either `Vector` or `Matrix`.
//!
//! * Level 1: `vector`
//! * Level 2: `matrix_vector`
//! * Level 3: `matrix`

extern crate core;
extern crate libc;
extern crate num;

pub use vector::Vector;
pub use vector::VectorOperations;
pub use matrix::Matrix;
pub use vector::ops::*;
pub use matrix_vector::ops::*;
pub use matrix::ops::*;
pub use transpose::Marker::{T, H};

#[macro_use]
mod prefix;
mod pointer;
mod scalar;

pub mod attribute;
pub mod default;
pub mod transpose;
pub mod vector;
pub mod matrix_vector;
pub mod matrix;
