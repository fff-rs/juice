// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

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

pub use crate::matrix::ops::*;
pub use crate::matrix::Matrix;
pub use crate::matrix_vector::ops::*;
pub use crate::vector::ops::*;
pub use crate::vector::Vector;
pub use crate::vector::VectorOperations;

#[macro_use]
mod prefix;
mod pointer;
mod scalar;

pub mod attribute;
pub mod default;
pub mod matrix;
pub mod matrix_vector;
pub mod vector;

pub mod math;
