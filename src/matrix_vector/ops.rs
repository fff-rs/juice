// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

extern crate num;

use self::num::complex::{Complex32, Complex64};
use matrix::{BlasMatrix};
use matrix_vector;
use pointer::CPtr;
use scalar::Scalar;
use vector::BlasVector;

pub trait Gemv {
    fn gemv(alpha: Self, a: &BlasMatrix<Self>, x: &BlasVector<Self>, beta: Self, y: &mut BlasVector<Self>);
}

macro_rules! gemv_impl(
    ($t: ty, $gemv_fn: ident) => (
        impl Gemv for $t {
            fn gemv(alpha: $t, a: &BlasMatrix<$t>, x: &BlasVector<$t>, beta: $t, y: &mut BlasVector<$t>){
                unsafe {
                    matrix_vector::ll::$gemv_fn(a.order(), a.transpose(),
                        a.rows(), a.cols(),
                        (&alpha).as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        x.as_ptr().as_c_ptr(), x.inc(),
                        (&beta).as_const(),
                        y.as_mut_ptr().as_c_ptr(), y.inc());
                }
            }
        }
    );
)

gemv_impl!(f32,       cblas_sgemv)
gemv_impl!(f64,       cblas_dgemv)
gemv_impl!(Complex32, cblas_cgemv)
gemv_impl!(Complex64, cblas_zgemv)
