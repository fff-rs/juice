// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

use num::complex::Complex;
use libc::{c_double, c_float, c_void};

pub trait Scalar<T, S> {
    fn as_const(self) -> T;
    fn as_mut(self) -> S;
}

macro_rules! scalar_impl(
    ($t: ty, $c_type: ty) => (
        impl<'a> Scalar<$t, *mut $t> for &'a $t {
            #[inline]
            fn as_const(self) -> $t {
                *self as $c_type
            }

            #[inline]
            fn as_mut(self) -> *mut $t {
                &self as *const _ as *mut $c_type
            }
        }

        impl<'a> Scalar<*const c_void, *mut c_void> for &'a Complex<$t> {
            #[inline]
            fn as_const(self) -> *const c_void {
                self as *const _ as *const c_void
            }

            #[inline]
            fn as_mut(self) -> *mut c_void {
                self as *const _ as *mut c_void
            }
        }
    );
);

scalar_impl!(f32, c_float);
scalar_impl!(f64, c_double);
