// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

extern crate num;

use self::num::complex::{Complex, Complex32, Complex64};
use std::cmp;
use pointer::CPtr;
use scalar::Scalar;
use vector;
use vector::BlasVector;

pub trait Copy {
    fn copy(x: &BlasVector<Self>, y: &mut BlasVector<Self>);
}

macro_rules! copy_impl(
    ($t: ty, $copy_fn: ident) => (
        impl Copy for $t {
            fn copy(x: &BlasVector<$t>, y: &mut BlasVector<$t>) {
                unsafe {
                    vector::ll::$copy_fn(x.len(),
                        x.as_ptr().as_c_ptr(),  x.inc(),
                        y.as_mut_ptr().as_c_ptr(), y.inc());
                }
            }
        }
    );
)

copy_impl!(f32,       cblas_scopy)
copy_impl!(f64,       cblas_dcopy)
copy_impl!(Complex32, cblas_ccopy)
copy_impl!(Complex64, cblas_zcopy)

pub trait Axpy {
    fn axpy(alpha: Self, x: &BlasVector<Self>, y: &mut BlasVector<Self>);
}

macro_rules! axpy_impl(
    ($t: ty, $update_fn: ident) => (
        impl Axpy for $t {
            fn axpy(alpha: $t, x: &BlasVector<$t>, y: &mut BlasVector<$t>) {
                unsafe {
                    let n = cmp::min(x.len(), y.len());

                    vector::ll::$update_fn(n,
                        (&alpha).as_const(),
                        x.as_ptr().as_c_ptr(), x.inc(),
                        y.as_mut_ptr().as_c_ptr(), y.inc());
                }
            }
        }
    );
)

axpy_impl!(f32,       cblas_saxpy)
axpy_impl!(f64,       cblas_daxpy)
axpy_impl!(Complex32, cblas_caxpy)
axpy_impl!(Complex64, cblas_zaxpy)

#[cfg(test)]
mod axpy_tests {
    extern crate num;
    extern crate test;

    use self::num::complex::Complex;
    use vector::ops::Axpy;

    #[test]
    fn real() {
        let x = vec![1f32,-2f32,3f32,4f32];
        let y = vec![3f32,7f32,-2f32,2f32];
        let mut z = y.clone();

        Axpy::axpy(1f32, &y, &mut z);
        Axpy::axpy(1f32, &x, &mut z);
        assert_eq!(z, vec![7f32,12f32,-1f32,8f32]);
    }

    #[test]
    fn complex() {
        let x = vec![Complex::new(1f32, 1f32), Complex::new(1f32, 3f32)];
        let y = vec![Complex::new(3f32, -2f32), Complex::new(2f32, 3f32)];
        let mut z = x.clone();

        Axpy::axpy(Complex::new(-1f32, 1f32), &y, &mut z);
        assert_eq!(z, vec![Complex::new(0f32, 6f32), Complex::new(-4f32, 2f32)]);
    }

}
