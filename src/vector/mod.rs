// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

//! Vector operations.
use libc::c_int;
use num::traits::NumCast;
use num::complex::{Complex32, Complex64};
use vector::ops::{Copy, Axpy, Scal, Dot, Nrm2, Asum, Iamax};

pub mod ll;
pub mod ops;

/// Methods that allow a type to be used in BLAS functions as a vector.
pub trait Vector<T> {
    /// The stride within the vector. For example, if `inc` returns 7, every
    /// 7th element is used. Defaults to 1.
    fn inc(&self) -> c_int { 1 }
    /// The number of elements in the vector.
    fn len(&self) -> c_int;
    /// An unsafe pointer to a contiguous block of memory.
    fn as_ptr(&self) -> *const T;
    /// An unsafe mutable pointer to a contiguous block of memory.
    fn as_mut_ptr(&mut self) -> *mut T;
}

impl<'a, T> Into<Vec<T>> for &'a Vector<T>
    where T: Copy {

    fn into(self) -> Vec<T> {
        let n = self.len() as usize;

        let mut x = Vec::with_capacity(n);
        unsafe { x.set_len(n); }
        Copy::copy(self, &mut x);

        x
    }
}

pub trait VectorOperations<T>: Sized + Vector<T>
    where T: Copy + Axpy + Scal + Dot + Nrm2 + Asum + Iamax {

    #[inline]
    fn update(&mut self, alpha: &T, x: &Vector<T>) -> &mut Self {
        Axpy::axpy(alpha, x, self);
        self
    }

    #[inline]
    fn scale(&mut self, alpha: &T) -> &mut Self {
        Scal::scal(alpha, self);
        self
    }

    #[inline]
    fn dot(&self, x: &Vector<T>) -> T {
        Dot::dot(self, x)
    }

    #[inline]
    fn abs_sum(&self) -> T {
        Asum::asum(self)
    }

    #[inline]
    fn norm(&self) -> T {
        Nrm2::nrm2(self)
    }

    #[inline]
    fn max_index(&self) -> usize {
        Iamax::iamax(self)
    }
}

impl<T> Vector<T> for Vec<T> {

    #[inline]
    fn len(&self) -> c_int {
        let l: Option<c_int> = NumCast::from(Vec::len(self));
        match l {
            Some(l) => l,
            None => panic!(),
        }
    }

    #[inline]
    fn as_ptr(&self) -> *const T { self[..].as_ptr() }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T { (&mut self[..]).as_mut_ptr() }
}

impl<T> Vector<T> for [T] {

    #[inline]
    fn len(&self) -> c_int {
        let l: Option<c_int> = NumCast::from(<[T]>::len(self));
        match l {
            Some(l) => l,
            None => panic!(),
        }
    }

    #[inline]
    fn as_ptr(&self) -> *const T { <[T]>::as_ptr(self) }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T { <[T]>::as_mut_ptr(self) }
}

macro_rules! operations_impl(
    ($v: ident, $($t: ty), +) => (
        $( impl VectorOperations<$t> for $v<$t> {} )+
    )
);

operations_impl!(Vec, f32, f64, Complex32, Complex64);
//impl<'a> VectorOperations<f32> for &'a [f32] {}
//impl<'a> VectorOperations<f64> for &'a [f64] {}
//impl<'a> VectorOperations<Complex32> for &'a [Complex32] {}
//impl<'a> VectorOperations<Complex64> for &'a [Complex64] {}
