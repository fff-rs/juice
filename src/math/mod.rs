// Copyright 2015 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

use std::ops::Deref;

pub use self::mat::Mat;

pub mod mat;
pub mod vector;
pub mod matrix_vector;

pub enum Trans<A> {
    T(A),
    H(A),
}

impl<A> Deref for Trans<A> {
    type Target = A;

    fn deref(&self) -> &A {
        match self {
            &Trans::T(ref v) => v,
            &Trans::H(ref v) => v,
        }
    }
}

pub enum Marker {
    T,
    H,
}
