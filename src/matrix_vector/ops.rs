// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

extern crate num;

use self::num::complex::{Complex32, Complex64};
use attribute::Symmetry;
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

pub trait Symv {
    fn symv(symmetry: Symmetry, alpha: Self, a: &BlasMatrix<Self>, x: &BlasVector<Self>, beta: Self, y: &mut BlasVector<Self>);
}

pub trait Hemv {
    fn hemv(symmetry: Symmetry, alpha: Self, a: &BlasMatrix<Self>, x: &BlasVector<Self>, beta: Self, y: &mut BlasVector<Self>);
}

macro_rules! symv_impl(
    ($trait_name: ident, $fn_name: ident, $t: ty, $symv_fn: ident) => (
        impl $trait_name for $t {
            fn $fn_name(symmetry: Symmetry, alpha: $t, a: &BlasMatrix<$t>, x: &BlasVector<$t>, beta: $t, y: &mut BlasVector<$t>){
                unsafe {
                    matrix_vector::ll::$symv_fn(a.order(), symmetry,
                        a.rows(),
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

symv_impl!(Symv, symv, f32,       cblas_ssymv)
symv_impl!(Symv, symv, f64,       cblas_dsymv)
symv_impl!(Symv, symv, Complex32, cblas_csymv)
symv_impl!(Symv, symv, Complex64, cblas_zsymv)

symv_impl!(Hemv, hemv, Complex32, cblas_chemv)
symv_impl!(Hemv, hemv, Complex64, cblas_zhemv)

pub trait Ger {
    fn ger(alpha: Self, x: &BlasVector<Self>, y: &BlasVector<Self>, a: &mut BlasMatrix<Self>);
}

pub trait Gerc {
    fn gerc(alpha: Self, x: &BlasVector<Self>, y: &BlasVector<Self>, a: &mut BlasMatrix<Self>);
}

macro_rules! ger_impl(
    ($trait_name: ident, $fn_name: ident, $t: ty, $ger_fn: ident) => (
        impl $trait_name for $t {
            fn $fn_name(alpha: $t, x: &BlasVector<$t>, y: &BlasVector<$t>, a: &mut BlasMatrix<$t>) {
                unsafe {
                    matrix_vector::ll::$ger_fn(a.order(),
                        a.rows(), a.cols(),
                        (&alpha).as_const(),
                        x.as_ptr().as_c_ptr(), x.inc(),
                        y.as_ptr().as_c_ptr(), y.inc(),
                        a.as_mut_ptr().as_c_ptr(), a.lead_dim());
                }
            }
        }
    );
)

ger_impl!(Ger, ger, f32,       cblas_sger)
ger_impl!(Ger, ger, f64,       cblas_dger)
ger_impl!(Ger, ger, Complex32, cblas_cgeru)
ger_impl!(Ger, ger, Complex64, cblas_zgeru)

ger_impl!(Gerc, gerc, Complex32, cblas_cgerc)
ger_impl!(Gerc, gerc, Complex64, cblas_zgerc)
