// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

extern crate num;

use self::num::complex::{Complex, Complex32, Complex64};
use attribute::{Diagonal, Side, Symmetry};
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

pub trait Symm {
    fn symm(side: Side, symmetry: Symmetry, alpha: Self, a: &BlasMatrix<Self>, b: &BlasMatrix<Self>, beta: Self, c: &mut BlasMatrix<Self>);
}

pub trait Hemm {
    fn hemm(side: Side, symmetry: Symmetry, alpha: Self, a: &BlasMatrix<Self>, b: &BlasMatrix<Self>, beta: Self, c: &mut BlasMatrix<Self>);
}

macro_rules! symm_impl(
    ($trait_name: ident, $fn_name: ident, $t: ty, $symm_fn: ident) => (
        impl $trait_name for $t {
            fn $fn_name(side: Side, symmetry: Symmetry, alpha: $t, a: &BlasMatrix<$t>, b: &BlasMatrix<$t>, beta: $t, c: &mut BlasMatrix<$t>) {
                unsafe {
                    matrix::ll::$symm_fn(a.order(),
                        side, symmetry,
                        a.rows(), b.cols(),
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

symm_impl!(Symm, symm, f32,       cblas_ssymm)
symm_impl!(Symm, symm, f64,       cblas_dsymm)
symm_impl!(Symm, symm, Complex32, cblas_csymm)
symm_impl!(Symm, symm, Complex64, cblas_zsymm)

symm_impl!(Hemm, hemm, Complex32, cblas_chemm)
symm_impl!(Hemm, hemm, Complex64, cblas_zhemm)

pub trait Trmm {
    fn trmm(side: Side, symmetry: Symmetry, diag: Diagonal, alpha: Self, a: &BlasMatrix<Self>, b: &mut BlasMatrix<Self>);
}

pub trait Trsm {
    fn trsm(side: Side, symmetry: Symmetry, diag: Diagonal, alpha: Self, a: &BlasMatrix<Self>, b: &mut BlasMatrix<Self>);
}

macro_rules! trmm_impl(
    ($trait_name: ident, $fn_name: ident, $t: ty, $symm_fn: ident) => (
        impl $trait_name for $t {
            fn $fn_name(side: Side, symmetry: Symmetry, diag: Diagonal, alpha: $t, a: &BlasMatrix<$t>, b: &mut BlasMatrix<$t>) {
                unsafe {
                    matrix::ll::$symm_fn(a.order(),
                        side, symmetry, a.transpose(), diag,
                        b.rows(), b.cols(),
                        (&alpha).as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        b.as_mut_ptr().as_c_ptr(), b.lead_dim());
                }
            }
        }
    );
)

trmm_impl!(Trmm, trmm, f32,       cblas_strmm)
trmm_impl!(Trmm, trmm, f64,       cblas_dtrmm)
trmm_impl!(Trmm, trmm, Complex32, cblas_ctrmm)
trmm_impl!(Trmm, trmm, Complex64, cblas_ztrmm)

trmm_impl!(Trsm, trsm, Complex32, cblas_ctrsm)
trmm_impl!(Trsm, trsm, Complex64, cblas_ztrsm)

pub trait Herk {
    fn herk(symmetry: Symmetry, alpha: Self, a: &BlasMatrix<Complex<Self>>, beta: Self, c: &mut BlasMatrix<Complex<Self>>);
}

macro_rules! herk_impl(
    ($t: ty, $symm_fn: ident) => (
        impl Herk for $t {
            fn herk(symmetry: Symmetry, alpha: $t, a: &BlasMatrix<Complex<$t>>, beta: $t, c: &mut BlasMatrix<Complex<$t>>) {
                unsafe {
                    matrix::ll::$symm_fn(a.order(),
                        symmetry, a.transpose(),
                        a.rows(), a.cols(),
                        alpha,
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        beta,
                        c.as_mut_ptr().as_c_ptr(), c.lead_dim());
                }
            }
        }
    );
)

herk_impl!(f32, cblas_cherk)
herk_impl!(f64, cblas_zherk)
