// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

extern crate num;

use self::num::complex::{Complex, Complex32, Complex64};
use attribute::{Diagonal, Symmetry};
use matrix::{BandMatrix, BlasMatrix};
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

pub trait Her {
    fn her(symmetry: Symmetry, alpha: Self, x: &BlasVector<Complex<Self>>, a: &mut BlasMatrix<Complex<Self>>);
}

macro_rules! her_impl(
    ($t: ty, $her_fn: ident) => (
        impl Her for $t {
            fn her(symmetry: Symmetry, alpha: $t, x: &BlasVector<Complex<$t>>, a: &mut BlasMatrix<Complex<$t>>) {
                unsafe {
                    matrix_vector::ll::$her_fn(a.order(), symmetry,
                        a.rows(),
                        alpha,
                        x.as_ptr().as_c_ptr(), x.inc(),
                        a.as_mut_ptr().as_c_ptr(), a.lead_dim());
                }
            }
        }
    );
)

her_impl!(f32, cblas_cher)
her_impl!(f64, cblas_zher)

pub trait Syr {
    fn syr(symmetry: Symmetry, alpha: Self, x: &BlasVector<Self>, a: &mut BlasMatrix<Self>);
}

macro_rules! syr_impl(
    ($t: ty, $syr_fn: ident) => (
        impl Syr for $t {
            fn syr(symmetry: Symmetry, alpha: $t, x: &BlasVector<$t>, a: &mut BlasMatrix<$t>) {
                unsafe {
                    matrix_vector::ll::$syr_fn(a.order(), symmetry,
                        a.rows(),
                        alpha,
                        x.as_ptr().as_c_ptr(), x.inc(),
                        a.as_mut_ptr().as_c_ptr(), a.lead_dim());
                }
            }
        }
    );
)

syr_impl!(f32, cblas_ssyr)
syr_impl!(f64, cblas_dsyr)

pub trait Syr2 {
    fn syr2(symmetry: Symmetry, alpha: Self, x: &BlasVector<Self>, y: &BlasVector<Self>, a: &mut BlasMatrix<Self>);
}

pub trait Her2 {
    fn her2(symmetry: Symmetry, alpha: Self, x: &BlasVector<Self>, y: &BlasVector<Self>, a: &mut BlasMatrix<Self>);
}

macro_rules! syr2_impl(
    ($trait_name: ident, $fn_name: ident, $t: ty, $syr2_fn: ident) => (
        impl $trait_name for $t {
            fn $fn_name(symmetry: Symmetry, alpha: $t, x: &BlasVector<$t>, y: &BlasVector<$t>, a: &mut BlasMatrix<$t>) {
                unsafe {
                    matrix_vector::ll::$syr2_fn(a.order(), symmetry,
                        a.rows(),
                        (&alpha).as_const(),
                        x.as_ptr().as_c_ptr(), x.inc(),
                        y.as_ptr().as_c_ptr(), y.inc(),
                        a.as_mut_ptr().as_c_ptr(), a.lead_dim());
                }
            }
        }
    );
)

syr2_impl!(Syr2, syr2, f32, cblas_ssyr2)
syr2_impl!(Syr2, syr2, f64, cblas_dsyr2)

syr2_impl!(Her2, her2, Complex32, cblas_cher2)
syr2_impl!(Her2, her2, Complex64, cblas_zher2)

pub trait Gbmv {
    fn gbmv(alpha: Self, a: &BandMatrix<Self>, x: &BlasVector<Self>, beta: Self, y: &mut BlasVector<Self>);
}

macro_rules! gbmv_impl(
    ($t: ty, $gbmv_fn: ident) => (
        impl Gbmv for $t {
            fn gbmv(alpha: $t, a: &BandMatrix<$t>, x: &BlasVector<$t>, beta: $t, y: &mut BlasVector<$t>){
                unsafe {
                    matrix_vector::ll::$gbmv_fn(a.order(), a.transpose(),
                        a.rows(), a.cols(),
                        a.sub_diagonals(), a.sup_diagonals(),
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

gbmv_impl!(f32,       cblas_sgbmv)
gbmv_impl!(f64,       cblas_dgbmv)
gbmv_impl!(Complex32, cblas_cgbmv)
gbmv_impl!(Complex64, cblas_zgbmv)

pub trait Sbmv {
    fn sbmv(symmetry: Symmetry, alpha: Self, a: &BandMatrix<Self>, x: &BlasVector<Self>, beta: Self, y: &mut BlasVector<Self>);
}

pub trait Hbmv {
    fn hbmv(symmetry: Symmetry, alpha: Self, a: &BandMatrix<Self>, x: &BlasVector<Self>, beta: Self, y: &mut BlasVector<Self>);
}

macro_rules! sbmv_impl(
    ($trait_name: ident, $fn_name: ident, $t: ty, $sbmv_fn: ident) => (
        impl $trait_name for $t {
            fn $fn_name(symmetry: Symmetry, alpha: $t, a: &BandMatrix<$t>, x: &BlasVector<$t>, beta: $t, y: &mut BlasVector<$t>) {
                unsafe {
                    matrix_vector::ll::$sbmv_fn(a.order(), symmetry,
                        a.rows(), a.sub_diagonals(),
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

sbmv_impl!(Sbmv, sbmv, f32,          cblas_ssbmv)
sbmv_impl!(Sbmv, sbmv, f64,          cblas_dsbmv)

sbmv_impl!(Hbmv, hbmv, Complex32, cblas_chbmv)
sbmv_impl!(Hbmv, hbmv, Complex64, cblas_zhbmv)

pub trait Tbmv {
    fn tbmv(symmetry: Symmetry, diagonal: Diagonal, a: &BandMatrix<Self>, x: &mut BlasVector<Self>);
}

pub trait Tbsv {
    fn tbsv(symmetry: Symmetry, diagonal: Diagonal, a: &BandMatrix<Self>, x: &mut BlasVector<Self>);
}

macro_rules! tbmv_impl(
    ($trait_name: ident, $fn_name: ident, $t: ty, $tbmv_fn: ident) => (
        impl $trait_name for $t {
            fn $fn_name(symmetry: Symmetry, diagonal: Diagonal, a: &BandMatrix<$t>, x: &mut BlasVector<$t>) {
                unsafe {
                    matrix_vector::ll::$tbmv_fn(a.order(), symmetry,
                        a.transpose(), diagonal,
                        a.rows(), a.sub_diagonals(),
                        a.as_ptr().as_c_ptr(),
                        x.as_mut_ptr().as_c_ptr(), x.inc());
                }
            }
        }
    );
)

tbmv_impl!(Tbmv, tbmv, f32,       cblas_stbmv)
tbmv_impl!(Tbmv, tbmv, f64,       cblas_dtbmv)
tbmv_impl!(Tbmv, tbmv, Complex32, cblas_ctbmv)
tbmv_impl!(Tbmv, tbmv, Complex64, cblas_ztbmv)

tbmv_impl!(Tbsv, tbsv, f32,       cblas_stbsv)
tbmv_impl!(Tbsv, tbsv, f64,       cblas_dtbsv)
tbmv_impl!(Tbsv, tbsv, Complex32, cblas_ctbsv)
tbmv_impl!(Tbsv, tbsv, Complex64, cblas_ztbsv)

pub trait Spmv {
    fn spmv(symmetry: Symmetry, alpha: Self, a: &BlasMatrix<Self>, x: &BlasVector<Self>, beta: Self, y: &mut BlasVector<Self>);
}

pub trait Hpmv {
    fn hpmv(symmetry: Symmetry, alpha: Self, a: &BlasMatrix<Self>, x: &BlasVector<Self>, beta: Self, y: &mut BlasVector<Self>);
}

macro_rules! spmv_impl(
    ($trait_name: ident, $fn_name: ident, $t: ty, $spmv_fn: ident) => (
        impl $trait_name for $t {
            fn $fn_name(symmetry: Symmetry, alpha: $t, a: &BlasMatrix<$t>, x: &BlasVector<$t>, beta: $t, y: &mut BlasVector<$t>) {
                unsafe {
                    matrix_vector::ll::$spmv_fn(a.order(), symmetry,
                        a.rows(),
                        (&alpha).as_const(),
                        a.as_ptr().as_c_ptr(),
                        x.as_ptr().as_c_ptr(), x.inc(),
                        (&beta).as_const(),
                        y.as_mut_ptr().as_c_ptr(), y.inc());
                }
            }
        }
    );
)

spmv_impl!(Spmv, spmv, f32,          cblas_sspmv)
spmv_impl!(Spmv, spmv, f64,          cblas_dspmv)

spmv_impl!(Hpmv, hpmv, Complex32, cblas_chpmv)
spmv_impl!(Hpmv, hpmv, Complex64, cblas_zhpmv)

pub trait Tpmv {
    fn tpmv(symmetry: Symmetry, diagonal: Diagonal, a: &BlasMatrix<Self>, x: &mut BlasVector<Self>);
}

pub trait Tpsv {
    fn tpsv(symmetry: Symmetry, diagonal: Diagonal, a: &BlasMatrix<Self>, x: &mut BlasVector<Self>);
}

macro_rules! tpmv_impl(
    ($trait_name: ident, $fn_name: ident, $t: ty, $tpmv_fn: ident) => (
        impl $trait_name for $t {
            fn $fn_name(symmetry: Symmetry, diagonal: Diagonal, a: &BlasMatrix<$t>, x: &mut BlasVector<$t>) {
                unsafe {
                    matrix_vector::ll::$tpmv_fn(a.order(), symmetry,
                        a.transpose(), diagonal,
                        a.rows(),
                        a.as_ptr().as_c_ptr(),
                        x.as_mut_ptr().as_c_ptr(), x.inc());
                }
            }
        }
    );
)

tpmv_impl!(Tpmv, tpmv, f32,          cblas_stpmv)
tpmv_impl!(Tpmv, tpmv, f64,          cblas_dtpmv)
tpmv_impl!(Tpmv, tpmv, Complex32, cblas_ctpmv)
tpmv_impl!(Tpmv, tpmv, Complex64, cblas_ztpmv)

tpmv_impl!(Tpsv, tpsv, f32,          cblas_stpsv)
tpmv_impl!(Tpsv, tpsv, f64,          cblas_dtpsv)
tpmv_impl!(Tpsv, tpsv, Complex32, cblas_ctpsv)
tpmv_impl!(Tpsv, tpsv, Complex64, cblas_ztpsv)

pub trait Hpr {
    fn hpr(symmetry: Symmetry, alpha: Self, x: &BlasVector<Complex<Self>>, a: &mut BlasMatrix<Complex<Self>>);
}

macro_rules! hpr_impl(
    ($t: ty, $hpr_fn: ident) => (
        impl Hpr for $t {
            fn hpr(symmetry: Symmetry, alpha: $t, x: &BlasVector<Complex<$t>>, a: &mut BlasMatrix<Complex<$t>>) {
                unsafe {
                    matrix_vector::ll::$hpr_fn(a.order(), symmetry,
                        a.rows(),
                        alpha,
                        x.as_ptr().as_c_ptr(), x.inc(),
                        a.as_mut_ptr().as_c_ptr());
                }
            }
        }
    );
)

hpr_impl!(f32, cblas_chpr)
hpr_impl!(f64, cblas_zhpr)

pub trait Spr {
    fn spr(symmetry: Symmetry, alpha: Self, x: &BlasVector<Self>, a: &mut BlasMatrix<Self>);
}

macro_rules! spr_impl(
    ($t: ty, $spr_fn: ident) => (
        impl Spr for $t {
            fn spr(symmetry: Symmetry, alpha: $t, x: &BlasVector<$t>, a: &mut BlasMatrix<$t>) {
                unsafe {
                    matrix_vector::ll::$spr_fn(a.order(), symmetry,
                        a.rows(),
                        alpha,
                        x.as_ptr().as_c_ptr(), x.inc(),
                        a.as_mut_ptr().as_c_ptr());
                }
            }
        }
    );
)

spr_impl!(f32, cblas_sspr)
spr_impl!(f64, cblas_dspr)
