// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

extern crate num;

use self::num::complex::{Complex32, Complex64};
use pointer::CPtr;
use scalar::Scalar;
use matrix;
use matrix::BlasMatrix;
use vector::BlasVector;

pub trait Gemm {
    fn gemm(alpha: Self, a: &BlasMatrix<Self>, b: &BlasMatrix<Self>, beta: Self, c: &mut BlasMatrix<Self>);
}

macro_rules! gemm_impl(
    ($t: ty, $gemm_fn: ident) => (
        impl Gemm for $t {
            fn gemm(alpha: $t, a: &BlasMatrix<$t>, b: &BlasMatrix<$t>, beta: $t, c: &mut BlasMatrix<$t>) {
                unsafe {
                    matrix::ll::$gemm_fn(a.order(),
                        a.transpose(), b.transpose(),
                        a.rows(), b.cols(), a.cols(),
                        (&alpha).as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        b.as_ptr().as_c_ptr(), b.lead_dim(),
                        (&beta).as_const(),
                        c.as_mut_ptr().as_c_ptr(), c.lead_dim());
                }
            }
        }
    );
)

gemm_impl!(f32,       cblas_sgemm)
gemm_impl!(f64,       cblas_dgemm)
gemm_impl!(Complex32, cblas_cgemm)
gemm_impl!(Complex64, cblas_zgemm)
