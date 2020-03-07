// Copyright 2015 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

use crate::matrix::Matrix;
use crate::vector::Vector;
use std::ops::{BitXor, Deref};

pub use self::mat::Mat;

pub mod mat;
pub mod bandmat;
pub mod matrix;
pub mod matrix_vector;
pub mod vector;

pub enum Trans<A> {
    T(A),
    H(A),
}

impl<A> Deref for Trans<A> {
    type Target = A;

    fn deref(&self) -> &A {
        match *self {
            Trans::T(ref v) => v,
            Trans::H(ref v) => v,
        }
    }
}

pub enum Marker {
    T,
    H,
}

impl<'a, T> BitXor<Marker> for &'a dyn Vector<T> {
    type Output = Trans<&'a dyn Vector<T>>;

    fn bitxor(self, m: Marker) -> Trans<&'a dyn Vector<T>> {
        match m {
            Marker::T => Trans::T(self),
            Marker::H => Trans::H(self),
        }
    }
}

impl<'a, T> BitXor<Marker> for &'a dyn Matrix<T> {
    type Output = Trans<&'a dyn Matrix<T>>;

    fn bitxor(self, m: Marker) -> Trans<&'a dyn Matrix<T>> {
        match m {
            Marker::T => Trans::T(self),
            Marker::H => Trans::H(self),
        }
    }
}
