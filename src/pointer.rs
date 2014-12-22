// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

extern crate libc;
extern crate num;

use self::num::complex::{Complex32, Complex64};
use self::libc::{c_double, c_int, c_float, c_long, c_void};

pub trait CPtr<T> {
    fn as_c_ptr(self) -> T;
}

macro_rules! c_ptr_impl(
    ($t: ty, $c_type: ty) => (
        impl CPtr<*const $c_type> for *const $t {
            #[inline]
            fn as_c_ptr(self) -> *const $c_type {
                self as *const $c_type
            }
        }

        impl CPtr<*mut $c_type> for *mut $t {
            #[inline]
            fn as_c_ptr(self) -> *mut $c_type {
                self as *mut $c_type
            }
        }
    );
);

c_ptr_impl!(i32, c_int);
c_ptr_impl!(i64, c_long);
c_ptr_impl!(f32, c_float);
c_ptr_impl!(f64, c_double);
c_ptr_impl!(Complex32, c_void);
c_ptr_impl!(Complex64, c_void);
