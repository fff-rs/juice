// Copyright 2015 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

use std::ops::{
    BitXor,
    Deref,
};
use matrix::Matrix;
use vector::Vector;

pub use self::mat::Mat;

pub mod mat;
pub mod vector;
pub mod matrix_vector;
pub mod matrix;

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

impl<'a, T> BitXor<Marker> for &'a Vector<T>
{
    type Output = Trans<&'a Vector<T>>;

    fn bitxor(self, m: Marker) -> Trans<&'a Vector<T>> {
        match m {
            Marker::T => Trans::T(self),
            Marker::H => Trans::H(self),
        }
    }
}

impl<'a, T> BitXor<Marker> for &'a Matrix<T>
{
    type Output = Trans<&'a Matrix<T>>;

    fn bitxor(self, m: Marker) -> Trans<&'a Matrix<T>> {
        match m {
            Marker::T => Trans::T(self),
            Marker::H => Trans::H(self),
        }
    }
}
