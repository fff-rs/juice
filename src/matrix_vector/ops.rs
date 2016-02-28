// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

//! Wrappers for matrix-vector functions.

use num::complex::{Complex, Complex32, Complex64};
use attribute::{Diagonal, Symmetry, Transpose};
use matrix::{BandMatrix, Matrix};
use matrix_vector::ll::*;
use pointer::CPtr;
use scalar::Scalar;
use vector::Vector;

/// General multiply with vector
///
/// A ← αA<sup>OP</sup>x + βy
pub trait Gemv: Sized {
    fn gemv<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(trans: Transpose, alpha: &Self, a: &Matrix<Self>, x: &V, beta: &Self, y: &mut W);
}

macro_rules! gemv_impl(($($t: ident), +) => (
    $(
        impl Gemv for $t {
            fn gemv<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(trans: Transpose, alpha: &$t, a: &Matrix<$t>, x: &V, beta: &$t, y: &mut W){
                unsafe {
                    prefix!($t, gemv)(a.order(), trans,
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
    use attribute::Transpose;
    use matrix::tests::M;
    use matrix_vector::ops::Gemv;

    #[test]
    fn real() {
        let a = M(2, 2, vec![1.0, -2.0, 2.0, -4.0]);
        let x = vec![2.0, 1.0];
        let mut y = vec![1.0, 2.0];
        let t = Transpose::NoTrans;

        Gemv::gemv(t, &1f32, &a, &x, &0f32, &mut y);

        assert_eq!(y, vec![0.0, 0.0]);
    }

    #[test]
    fn non_square() {
        let a = M(2, 3,
                 vec![
                 1.0, -3.0, 1.0,
                 2.0, -6.0, 2.0]);
        let x = vec![2.0, 1.0, 1.0];
        let mut y = vec![1.0, 2.0];
        let t = Transpose::NoTrans;

        Gemv::gemv(t, &1f32, &a, &x, &0f32, &mut y);
        assert_eq!(y, vec![0.0, 0.0]);
    }

    #[test]
    fn transpose() {
        let a = M(3, 2,
                 vec![
                     1.0, 2.0,
                     -3.0, -6.0,
                     1.0, 2.0]);

        let x = vec![2.0, 1.0, 1.0];
        let mut y = vec![1.0, 2.0];
        let t = Transpose::Trans;

        Gemv::gemv(t, &1f32, &a, &x, &0f32, &mut y);

        assert_eq!(y, vec![0.0, 0.0]);
    }
}

/// Symmetric multiply with vector
///
/// A ← αAx + βy
pub trait Symv: Sized {
    fn symv<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(symmetry: Symmetry, alpha: &Self, a: &Matrix<Self>, x: &V, beta: &Self, y: &mut W);
}

/// Hermitian multiply with vector
///
/// A ← αAx + βy
pub trait Hemv: Sized {
    fn hemv<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(symmetry: Symmetry, alpha: &Self, a: &Matrix<Self>, x: &V, beta: &Self, y: &mut W);
}

macro_rules! symv_impl(($trait_name: ident, $fn_name: ident, $($t: ident), +) => (
    $(
        impl $trait_name for $t {
            fn $fn_name<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(symmetry: Symmetry, alpha: &$t, a: &Matrix<$t>, x: &V, beta: &$t, y: &mut W){
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

#[cfg(test)]
mod symv_tests {
    use attribute::{Symmetry, Transpose};
    use matrix::tests::M;
    use matrix_vector::ops::{Gemv, Symv};

    #[test]
    fn real() {
        let x = vec![2.0, 1.0];
        let gemv = {
            let a = M(2, 2, vec![1.0, -2.0, -2.0, 1.0]);
            let mut y = vec![1.0, 2.0];
            let t = Transpose::NoTrans;

            Gemv::gemv(t, &1.0, &a, &x, &0.0, &mut y);
            y
        };

        let symv_upper = {
            // symv shouldn't look at some elements
            let a = M(2, 2, vec![1.0, -2.0, 0.0, 1.0]);
            let mut y = vec![1.0, 2.0];
            let s = Symmetry::Upper;

            Symv::symv(s, &1.0, &a, &x, &0.0, &mut y);
            y
        };

        let symv_lower = {
            // symv shouldn't look at some elements
            let a = M(2, 2, vec![1.0, 0.0, -2.0, 1.0]);
            let mut y = vec![1.0, 2.0];
            let s = Symmetry::Lower;

            Symv::symv(s, &1.0, &a, &x, &0.0, &mut y);
            y
        };

        assert_eq!(gemv, symv_upper);
        assert_eq!(gemv, symv_lower);
    }
}

/// General rank-1 update
///
/// A ← A + αxy<sup>T</sup>
pub trait Ger: Sized {
    fn ger<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(alpha: &Self, x: &V, y: &W, a: &mut Matrix<Self>);
}

/// General rank-1 update (using hermitian conjugate)
///
/// A ← A + αxy<sup>H</sup>
pub trait Gerc: Ger {
    fn gerc<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(alpha: &Self, x: &V, y: &W, a: &mut Matrix<Self>) {
        Ger::ger(alpha, x, y, a);
    }
}

macro_rules! ger_impl(
    ($trait_name: ident, $fn_name: ident, $t: ty, $ger_fn: expr) => (
        impl $trait_name for $t {
            fn $fn_name<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(alpha: &$t, x: &V, y: &W, a: &mut Matrix<$t>) {
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

ger_impl!(Ger, ger, f32,       cblas_s::ger);
ger_impl!(Ger, ger, f64,       cblas_d::ger);
ger_impl!(Ger, ger, Complex32, cblas_c::geru);
ger_impl!(Ger, ger, Complex64, cblas_z::geru);

impl Gerc for f32 {}
impl Gerc for f64 {}
ger_impl!(Gerc, gerc, Complex32, cblas_c::gerc);
ger_impl!(Gerc, gerc, Complex64, cblas_z::gerc);

#[cfg(test)]
mod ger_tests {
    use std::iter::repeat;
    use matrix::tests::M;
    use matrix_vector::ops::Ger;

    #[test]
    fn real() {
        let mut a = M(3, 3, repeat(0.0).take(9).collect());
        let x = vec![2.0, 1.0, 4.0];
        let y = vec![3.0, 6.0, -1.0];

        Ger::ger(&1f32, &x, &y, &mut a);

        let result = vec![6.0, 12.0, -2.0, 3.0, 6.0, -1.0, 12.0, 24.0, -4.0];
        assert_eq!(a.2, result);
    }
}

/// Symmetric rank-1 update
///
/// A ← A + αxx<sup>T</sup>
pub trait Syr: Sized {
    fn syr<V: ?Sized + Vector<Self>>(symmetry: Symmetry, alpha: &Self, x: &V, a: &mut Matrix<Self>);
}

/// Hermitian rank-1 update
///
/// A ← A + αxx<sup>H</sup>
pub trait Her: Sized {
    fn her<V: ?Sized + Vector<Complex<Self>>>(symmetry: Symmetry, alpha: &Self, x: &V, a: &mut Matrix<Complex<Self>>);
}

macro_rules! her_impl(($($t: ident), +) => (
    $(
        impl Her for $t {
            fn her<V: ?Sized + Vector<Complex<Self>>>(symmetry: Symmetry, alpha: &$t, x: &V, a: &mut Matrix<Complex<$t>>) {
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
            fn syr<V: ?Sized + Vector<Self>>(symmetry: Symmetry, alpha: &$t, x: &V, a: &mut Matrix<$t>) {
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

/// Symmetric rank-2 update
///
/// A ← A + αxy<sup>T</sup> + αyx<sup>T</sup>
pub trait Syr2: Sized {
    fn syr2<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(symmetry: Symmetry, alpha: &Self, x: &V, y: &W, a: &mut Matrix<Self>);
}

/// Hermitian rank-2 update
///
/// A ← A + αxy<sup>H</sup> + y(αx)<sup>H</sup>
pub trait Her2: Sized {
    fn her2<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(symmetry: Symmetry, alpha: &Self, x: &V, y: &W, a: &mut Matrix<Self>);
}

macro_rules! syr2_impl(($trait_name: ident, $fn_name: ident, $($t: ident), +) => (
    $(
        impl $trait_name for $t {
            fn $fn_name<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(symmetry: Symmetry, alpha: &$t, x: &V, y: &W, a: &mut Matrix<$t>) {
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

/// General band matrix multiply with vector.
///
/// A ← αA<sup>OP</sup>x + βy
pub trait Gbmv: Sized {
    fn gbmv<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(trans: Transpose, alpha: &Self, a: &BandMatrix<Self>, x: &V, beta: &Self, y: &mut W);
}

macro_rules! gbmv_impl(($($t: ident), +) => (
    $(
        impl Gbmv for $t {
            fn gbmv<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(trans: Transpose, alpha: &$t, a: &BandMatrix<$t>, x: &V, beta: &$t, y: &mut W){
                unsafe {
                    prefix!($t, gbmv)(a.order(), trans,
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

/// Symmetric band matrix multiply with vector
///
/// A ← αAx + βy
pub trait Sbmv: Sized {
    fn sbmv<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(symmetry: Symmetry, alpha: &Self, a: &BandMatrix<Self>, x: &V, beta: &Self, y: &mut W);
}

/// Hermitian band matrix multiply with vector
///
/// A ← αAx + βy
pub trait Hbmv: Sized {
    fn hbmv<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(symmetry: Symmetry, alpha: &Self, a: &BandMatrix<Self>, x: &V, beta: &Self, y: &mut W);
}

macro_rules! sbmv_impl(($trait_name: ident, $fn_name: ident, $($t: ident), +) => (
    $(
        impl $trait_name for $t {
            fn $fn_name<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(symmetry: Symmetry, alpha: &$t, a: &BandMatrix<$t>, x: &V, beta: &$t, y: &mut W) {
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

/// Triangular band matrix multiply with vector
///
/// A ← A<sup>OP</sup>x
pub trait Tbmv: Sized {
    fn tbmv<V: ?Sized + Vector<Self>>(symmetry: Symmetry, trans: Transpose, diagonal: Diagonal, a: &BandMatrix<Self>, x: &mut V);
}

/// Solve triangular band matrix system
///
/// A ← A<sup>-1 OP</sup>x
pub trait Tbsv: Sized {
    fn tbsv<V: ?Sized + Vector<Self>>(symmetry: Symmetry, trans: Transpose, diagonal: Diagonal, a: &BandMatrix<Self>, x: &mut V);
}

macro_rules! tbmv_impl(($trait_name: ident, $fn_name: ident, $($t: ident), +) => (
    $(
        impl $trait_name for $t {
            fn $fn_name<V: ?Sized + Vector<Self>>(symmetry: Symmetry, trans: Transpose, diagonal: Diagonal, a: &BandMatrix<$t>, x: &mut V) {
                unsafe {
                    prefix!($t, $fn_name)(a.order(), symmetry,
                        trans, diagonal,
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

/// Symmetric packed matrix multiply with vector
///
/// A ← αAx + βy
pub trait Spmv: Sized {
    fn spmv<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(symmetry: Symmetry, alpha: &Self, a: &Matrix<Self>, x: &V, beta: &Self, y: &mut W);
}

/// Hermitian packed matrix multiply with vector
///
/// A ← αAx + βy
pub trait Hpmv: Sized {
    fn hpmv<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(symmetry: Symmetry, alpha: &Self, a: &Matrix<Self>, x: &V, beta: &Self, y: &mut W);
}

macro_rules! spmv_impl(($trait_name: ident, $fn_name: ident, $($t: ident), +) => (
    $(
        impl $trait_name for $t {
            fn $fn_name<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(symmetry: Symmetry, alpha: &$t, a: &Matrix<$t>, x: &V, beta: &$t, y: &mut W) {
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

/// Triangular packed matrix multiply with vector
///
/// A ← A<sup>OP</sup>x
pub trait Tpmv: Sized {
    fn tpmv<V: ?Sized + Vector<Self>>(symmetry: Symmetry, trans: Transpose, diagonal: Diagonal, a: &Matrix<Self>, x: &mut V);
}

/// Solve triangular packed matrix system
///
/// A ← A<sup>-1 OP</sup>x
pub trait Tpsv: Sized {
    fn tpsv<V: ?Sized + Vector<Self>>(symmetry: Symmetry, trans: Transpose, diagonal: Diagonal, a: &Matrix<Self>, x: &mut V);
}

macro_rules! tpmv_impl(($trait_name: ident, $fn_name: ident, $($t: ident), +) => (
    $(
        impl $trait_name for $t {
            fn $fn_name<V: ?Sized + Vector<Self>>(symmetry: Symmetry, trans: Transpose, diagonal: Diagonal, a: &Matrix<$t>, x: &mut V) {
                unsafe {
                    prefix!($t, $fn_name)(a.order(), symmetry,
                        trans, diagonal,
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

/// Hermitian packed matrix rank-1 update
///
/// A ← A + αxx<sup>H</sup>
pub trait Hpr: Sized {
    fn hpr<V: ?Sized + Vector<Complex<Self>>>(symmetry: Symmetry, alpha: &Self, x: &V, a: &mut Matrix<Complex<Self>>);
}

macro_rules! hpr_impl(($($t: ident), +) => (
    $(
        impl Hpr for $t {
            fn hpr<V: ?Sized + Vector<Complex<Self>>>(symmetry: Symmetry, alpha: &$t, x: &V, a: &mut Matrix<Complex<$t>>) {
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

/// Symmetric packed matrix rank-1 update
///
/// A ← A + αxx<sup>T</sup>
pub trait Spr: Sized {
    fn spr<V: ?Sized + Vector<Self>>(symmetry: Symmetry, alpha: &Self, x: &V, a: &mut Matrix<Self>);
}

macro_rules! spr_impl(($($t: ident), +) => (
    $(
        impl Spr for $t {
            fn spr<V: ?Sized + Vector<Self>>(symmetry: Symmetry, alpha: &$t, x: &V, a: &mut Matrix<$t>) {
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

/// Symmetric packed matrix rank-2 update
///
/// A ← A + αxy<sup>T</sup> + αyx<sup>T</sup>
pub trait Spr2: Sized {
    fn spr2<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(symmetry: Symmetry, alpha: &Self, x: &V, y: &W, a: &mut Matrix<Self>);
}

/// Hermitian packed matrix rank-2 update
///
/// A ← A + αxy<sup>H</sup> + y(αx)<sup>H</sup>
pub trait Hpr2: Sized {
    fn hpr2<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(symmetry: Symmetry, alpha: &Self, x: &V, y: &W, a: &mut Matrix<Self>);
}

macro_rules! spr2_impl(($trait_name: ident, $fn_name: ident, $($t: ident), +) => (
    $(
        impl $trait_name for $t {
            fn $fn_name<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(symmetry: Symmetry, alpha: &$t, x: &V, y: &W, a: &mut Matrix<$t>) {
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
