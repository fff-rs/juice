// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.
#![feature(tuple_indexing)]
#![feature(macro_rules)]

pub use vector::Vector;
pub use vector::VectorOperations;
pub use matrix::Matrix;

pub mod mat;

pub mod vector;
pub mod matrix_vector;
pub mod matrix;

pub mod attribute;
pub mod default;
pub mod pointer;
mod scalar;
