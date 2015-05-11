// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

//! Wrappers for matrix-vector functions.

use num::complex::{Complex, Complex32, Complex64};
use attribute::{Diagonal, Symmetry};
use matrix::{BandMatrix, Matrix};
use matrix_vector::ll::*;
use pointer::CPtr;
use scalar::Scalar;
use vector::Vector;

pub trait Gemv {
    fn gemv(alpha: &Self, a: &Matrix<Self>, x: &Vector<Self>, beta: &Self, y: &mut Vector<Self>);
}

macro_rules! gemv_impl(($($t: ident), +) => (
    $(
        impl Gemv for $t {
            fn gemv(alpha: &$t, a: &Matrix<$t>, x: &Vector<$t>, beta: &$t, y: &mut Vector<$t>){
                unsafe {
                    prefix!($t, gemv)(a.order(), a.transpose(),
                        a.rows(), a.cols(),
                        alpha.as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        x.as_ptr().as_c_ptr(), x.inc(),
                        beta.as_const(),
                        y.as_mut_ptr().as_c_ptr(), y.inc());
                }
            }
        }
    )+
));

gemv_impl!(f32, f64, Complex32, Complex64);

#[cfg(test)]
mod gemv_tests {
    use matrix_vector::ops::Gemv;

    #[test]
    fn real() {
        let a = (2, 2, vec![1.0, -2.0, 2.0, -4.0]);
        let x = vec![2.0, 1.0];
        let mut y = vec![1.0, 2.0];

        Gemv::gemv(&1f32, &a, &x, &0f32, &mut y);

        assert_eq!(y, vec![0.0, 0.0]);
    }
}

pub trait Symv {
    fn symv(symmetry: Symmetry, alpha: &Self, a: &Matrix<Self>, x: &Vector<Self>, beta: &Self, y: &mut Vector<Self>);
}

pub trait Hemv {
    fn hemv(symmetry: Symmetry, alpha: &Self, a: &Matrix<Self>, x: &Vector<Self>, beta: &Self, y: &mut Vector<Self>);
}

macro_rules! symv_impl(($trait_name: ident, $fn_name: ident, $($t: ident), +) => (
    $(
        impl $trait_name for $t {
            fn $fn_name(symmetry: Symmetry, alpha: &$t, a: &Matrix<$t>, x: &Vector<$t>, beta: &$t, y: &mut Vector<$t>){
                unsafe {
                    prefix!($t, $fn_name)(a.order(), symmetry,
                        a.rows(),
                        alpha.as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        x.as_ptr().as_c_ptr(), x.inc(),
                        beta.as_const(),
                        y.as_mut_ptr().as_c_ptr(), y.inc());
                }
            }
        }
    )+
));

symv_impl!(Symv, symv, f32, f64, Complex32, Complex64);
symv_impl!(Hemv, hemv, Complex32, Complex64);

pub trait Ger {
    fn ger(alpha: &Self, x: &Vector<Self>, y: &Vector<Self>, a: &mut Matrix<Self>);
}

pub trait Gerc: Ger {
    fn gerc(alpha: &Self, x: &Vector<Self>, y: &Vector<Self>, a: &mut Matrix<Self>) {
        Ger::ger(alpha, x, y, a);
    }
}

macro_rules! ger_impl(
    ($trait_name: ident, $fn_name: ident, $t: ty, $ger_fn: ident) => (
        impl $trait_name for $t {
            fn $fn_name(alpha: &$t, x: &Vector<$t>, y: &Vector<$t>, a: &mut Matrix<$t>) {
                unsafe {
                    $ger_fn(a.order(),
                        a.rows(), a.cols(),
                        alpha.as_const(),
                        x.as_ptr().as_c_ptr(), x.inc(),
                        y.as_ptr().as_c_ptr(), y.inc(),
                        a.as_mut_ptr().as_c_ptr(), a.lead_dim());
                }
            }
        }
    );
);

ger_impl!(Ger, ger, f32,       cblas_sger);
ger_impl!(Ger, ger, f64,       cblas_dger);
ger_impl!(Ger, ger, Complex32, cblas_cgeru);
ger_impl!(Ger, ger, Complex64, cblas_zgeru);

ger_impl!(Gerc, gerc, Complex32, cblas_cgerc);
ger_impl!(Gerc, gerc, Complex64, cblas_zgerc);

#[cfg(test)]
mod ger_tests {
    use std::iter::repeat;
    use matrix_vector::ops::Ger;

    #[test]
    fn real() {
        let mut a = (3, 3, repeat(0.0).take(9).collect());
        let x = vec![2.0, 1.0, 4.0];
        let y = vec![3.0, 6.0, -1.0];

        Ger::ger(&1f32, &x, &y, &mut a);

        let result = vec![6.0, 12.0, -2.0, 3.0, 6.0, -1.0, 12.0, 24.0, -4.0];
        assert_eq!(a.2, result);
    }
}

pub trait Syr {
    fn syr(symmetry: Symmetry, alpha: &Self, x: &Vector<Self>, a: &mut Matrix<Self>);
}

pub trait Her {
    fn her(symmetry: Symmetry, alpha: &Self, x: &Vector<Complex<Self>>, a: &mut Matrix<Complex<Self>>);
}

macro_rules! her_impl(($($t: ident), +) => (
    $(
        impl Her for $t {
            fn her(symmetry: Symmetry, alpha: &$t, x: &Vector<Complex<$t>>, a: &mut Matrix<Complex<$t>>) {
                unsafe {
                    prefix!(Complex<$t>, her)(a.order(), symmetry,
                        a.rows(),
                        *alpha,
                        x.as_ptr().as_c_ptr(), x.inc(),
                        a.as_mut_ptr().as_c_ptr(), a.lead_dim());
                }
            }
        }
    )+
));

her_impl!(f32, f64);

macro_rules! syr_impl(($($t: ident), +) => (
    $(
        impl Syr for $t {
            fn syr(symmetry: Symmetry, alpha: &$t, x: &Vector<$t>, a: &mut Matrix<$t>) {
                unsafe {
                    prefix!($t, syr)(a.order(), symmetry,
                        a.rows(),
                        *alpha,
                        x.as_ptr().as_c_ptr(), x.inc(),
                        a.as_mut_ptr().as_c_ptr(), a.lead_dim());
                }
            }
        }
    )+
));

syr_impl!(f32, f64);

pub trait Syr2 {
    fn syr2(symmetry: Symmetry, alpha: &Self, x: &Vector<Self>, y: &Vector<Self>, a: &mut Matrix<Self>);
}

pub trait Her2 {
    fn her2(symmetry: Symmetry, alpha: &Self, x: &Vector<Self>, y: &Vector<Self>, a: &mut Matrix<Self>);
}

macro_rules! syr2_impl(($trait_name: ident, $fn_name: ident, $($t: ident), +) => (
    $(
        impl $trait_name for $t {
            fn $fn_name(symmetry: Symmetry, alpha: &$t, x: &Vector<$t>, y: &Vector<$t>, a: &mut Matrix<$t>) {
                unsafe {
                    prefix!($t, $fn_name)(a.order(), symmetry,
                        a.rows(),
                        alpha.as_const(),
                        x.as_ptr().as_c_ptr(), x.inc(),
                        y.as_ptr().as_c_ptr(), y.inc(),
                        a.as_mut_ptr().as_c_ptr(), a.lead_dim());
                }
            }
        }
    )+
));

syr2_impl!(Syr2, syr2, f32, f64);
syr2_impl!(Her2, her2, Complex32, Complex64);

pub trait Gbmv {
    fn gbmv(alpha: &Self, a: &BandMatrix<Self>, x: &Vector<Self>, beta: &Self, y: &mut Vector<Self>);
}

macro_rules! gbmv_impl(($($t: ident), +) => (
    $(
        impl Gbmv for $t {
            fn gbmv(alpha: &$t, a: &BandMatrix<$t>, x: &Vector<$t>, beta: &$t, y: &mut Vector<$t>){
                unsafe {
                    prefix!($t, gbmv)(a.order(), a.transpose(),
                        a.rows(), a.cols(),
                        a.sub_diagonals(), a.sup_diagonals(),
                        alpha.as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        x.as_ptr().as_c_ptr(), x.inc(),
                        beta.as_const(),
                        y.as_mut_ptr().as_c_ptr(), y.inc());
                }
            }
        }
    )+
));

gbmv_impl!(f32, f64, Complex32, Complex64);

pub trait Sbmv {
    fn sbmv(symmetry: Symmetry, alpha: &Self, a: &BandMatrix<Self>, x: &Vector<Self>, beta: &Self, y: &mut Vector<Self>);
}

pub trait Hbmv {
    fn hbmv(symmetry: Symmetry, alpha: &Self, a: &BandMatrix<Self>, x: &Vector<Self>, beta: &Self, y: &mut Vector<Self>);
}

macro_rules! sbmv_impl(($trait_name: ident, $fn_name: ident, $($t: ident), +) => (
    $(
        impl $trait_name for $t {
            fn $fn_name(symmetry: Symmetry, alpha: &$t, a: &BandMatrix<$t>, x: &Vector<$t>, beta: &$t, y: &mut Vector<$t>) {
                unsafe {
                    prefix!($t, $fn_name)(a.order(), symmetry,
                        a.rows(), a.sub_diagonals(),
                        alpha.as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        x.as_ptr().as_c_ptr(), x.inc(),
                        beta.as_const(),
                        y.as_mut_ptr().as_c_ptr(), y.inc());
                }
            }
        }
    )+
));

sbmv_impl!(Sbmv, sbmv, f32, f64);
sbmv_impl!(Hbmv, hbmv, Complex32, Complex64);

pub trait Tbmv {
    fn tbmv(symmetry: Symmetry, diagonal: Diagonal, a: &BandMatrix<Self>, x: &mut Vector<Self>);
}

pub trait Tbsv {
    fn tbsv(symmetry: Symmetry, diagonal: Diagonal, a: &BandMatrix<Self>, x: &mut Vector<Self>);
}

macro_rules! tbmv_impl(($trait_name: ident, $fn_name: ident, $($t: ident), +) => (
    $(
        impl $trait_name for $t {
            fn $fn_name(symmetry: Symmetry, diagonal: Diagonal, a: &BandMatrix<$t>, x: &mut Vector<$t>) {
                unsafe {
                    prefix!($t, $fn_name)(a.order(), symmetry,
                        a.transpose(), diagonal,
                        a.rows(), a.sub_diagonals(),
                        a.as_ptr().as_c_ptr(),
                        x.as_mut_ptr().as_c_ptr(), x.inc());
                }
            }
        }
    )+
));

tbmv_impl!(Tbmv, tbmv, f32, f64, Complex32, Complex64);
tbmv_impl!(Tbsv, tbsv, f32, f64, Complex32, Complex64);

pub trait Spmv {
    fn spmv(symmetry: Symmetry, alpha: &Self, a: &Matrix<Self>, x: &Vector<Self>, beta: &Self, y: &mut Vector<Self>);
}

pub trait Hpmv {
    fn hpmv(symmetry: Symmetry, alpha: &Self, a: &Matrix<Self>, x: &Vector<Self>, beta: &Self, y: &mut Vector<Self>);
}

macro_rules! spmv_impl(($trait_name: ident, $fn_name: ident, $($t: ident), +) => (
    $(
        impl $trait_name for $t {
            fn $fn_name(symmetry: Symmetry, alpha: &$t, a: &Matrix<$t>, x: &Vector<$t>, beta: &$t, y: &mut Vector<$t>) {
                unsafe {
                    prefix!($t, $fn_name)(a.order(), symmetry,
                        a.rows(),
                        alpha.as_const(),
                        a.as_ptr().as_c_ptr(),
                        x.as_ptr().as_c_ptr(), x.inc(),
                        beta.as_const(),
                        y.as_mut_ptr().as_c_ptr(), y.inc());
                }
            }
        }
    )+
));

spmv_impl!(Spmv, spmv, f32, f64);
spmv_impl!(Hpmv, hpmv, Complex32, Complex64);

pub trait Tpmv {
    fn tpmv(symmetry: Symmetry, diagonal: Diagonal, a: &Matrix<Self>, x: &mut Vector<Self>);
}

pub trait Tpsv {
    fn tpsv(symmetry: Symmetry, diagonal: Diagonal, a: &Matrix<Self>, x: &mut Vector<Self>);
}

macro_rules! tpmv_impl(($trait_name: ident, $fn_name: ident, $($t: ident), +) => (
    $(
        impl $trait_name for $t {
            fn $fn_name(symmetry: Symmetry, diagonal: Diagonal, a: &Matrix<$t>, x: &mut Vector<$t>) {
                unsafe {
                    prefix!($t, $fn_name)(a.order(), symmetry,
                        a.transpose(), diagonal,
                        a.rows(),
                        a.as_ptr().as_c_ptr(),
                        x.as_mut_ptr().as_c_ptr(), x.inc());
                }
            }
        }
    )+
));

tpmv_impl!(Tpmv, tpmv, f32, f64, Complex32, Complex64);
tpmv_impl!(Tpsv, tpsv, f32, f64, Complex32, Complex64);

pub trait Hpr {
    fn hpr(symmetry: Symmetry, alpha: &Self, x: &Vector<Complex<Self>>, a: &mut Matrix<Complex<Self>>);
}

macro_rules! hpr_impl(($($t: ident), +) => (
    $(
        impl Hpr for $t {
            fn hpr(symmetry: Symmetry, alpha: &$t, x: &Vector<Complex<$t>>, a: &mut Matrix<Complex<$t>>) {
                unsafe {
                    prefix!(Complex<$t>, hpr)(a.order(), symmetry,
                        a.rows(),
                        *alpha,
                        x.as_ptr().as_c_ptr(), x.inc(),
                        a.as_mut_ptr().as_c_ptr());
                }
            }
        }
    )+
));

hpr_impl!(f32, f64);

pub trait Spr {
    fn spr(symmetry: Symmetry, alpha: &Self, x: &Vector<Self>, a: &mut Matrix<Self>);
}

macro_rules! spr_impl(($($t: ident), +) => (
    $(
        impl Spr for $t {
            fn spr(symmetry: Symmetry, alpha: &$t, x: &Vector<$t>, a: &mut Matrix<$t>) {
                unsafe {
                    prefix!($t, spr)(a.order(), symmetry,
                        a.rows(),
                        *alpha,
                        x.as_ptr().as_c_ptr(), x.inc(),
                        a.as_mut_ptr().as_c_ptr());
                }
            }
        }
    )+
));

spr_impl!(f32, f64);

pub trait Spr2 {
    fn spr2(symmetry: Symmetry, alpha: &Self, x: &Vector<Self>, y: &Vector<Self>, a: &mut Matrix<Self>);
}

pub trait Hpr2 {
    fn hpr2(symmetry: Symmetry, alpha: &Self, x: &Vector<Self>, y: &Vector<Self>, a: &mut Matrix<Self>);
}

macro_rules! spr2_impl(($trait_name: ident, $fn_name: ident, $($t: ident), +) => (
    $(
        impl $trait_name for $t {
            fn $fn_name(symmetry: Symmetry, alpha: &$t, x: &Vector<$t>, y: &Vector<$t>, a: &mut Matrix<$t>) {
                unsafe {
                    prefix!($t, $fn_name)(a.order(), symmetry,
                        a.rows(),
                        alpha.as_const(),
                        x.as_ptr().as_c_ptr(), x.inc(),
                        y.as_ptr().as_c_ptr(), y.inc(),
                        a.as_mut_ptr().as_c_ptr());
                }
            }
        }
    )+
));

spr2_impl!(Spr2, spr2, f32, f64);
spr2_impl!(Hpr2, hpr2, Complex32, Complex64);
