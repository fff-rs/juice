// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.
#![feature(concat_idents)]
#![feature(core)]
#![feature(libc)]

extern crate core;
extern crate libc;
extern crate num;

pub use vector::Vector;
pub use vector::VectorOperations;
pub use matrix::Matrix;
pub use vector::ops::*;
pub use matrix_vector::ops::*;
pub use matrix::ops::*;

#[macro_use]
mod prefix;
mod default;
mod pointer;
mod scalar;

#[stable]
pub mod attribute;
#[unstable]
pub mod vector;
#[unstable]
pub mod matrix_vector;
#[unstable]
pub mod matrix;
