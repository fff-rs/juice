// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

use num::complex::{Complex, Complex32, Complex64};
use attribute::{Diagonal, Side, Symmetry};
use pointer::CPtr;
use scalar::Scalar;
use matrix::ll::*;
use matrix::Matrix;
use vector::Vector;

pub trait Gemm {
    fn gemm(alpha: &Self, a: &Matrix<Self>, b: &Matrix<Self>, beta: &Self, c: &mut Matrix<Self>);
}

macro_rules! gemm_impl(($($t: ident), +) => (
    $(
        impl Gemm for $t {
            fn gemm(alpha: &$t, a: &Matrix<$t>, b: &Matrix<$t>, beta: &$t, c: &mut Matrix<$t>) {
                unsafe {
                    prefix!($t, gemm)(a.order(),
                        a.transpose(), b.transpose(),
                        a.rows(), b.cols(), a.cols(),
                        alpha.as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        b.as_ptr().as_c_ptr(), b.lead_dim(),
                        beta.as_const(),
                        c.as_mut_ptr().as_c_ptr(), c.lead_dim());
                }
            }
        }
    )+
));

gemm_impl!(f32, f64, Complex32, Complex64);

#[cfg(test)]
mod gemm_tests {
    use std::iter::repeat;
    use matrix::ops::Gemm;

    #[test]
    fn real() {
        let a = (2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = (2, 2, vec![-1.0, 3.0, 1.0, 1.0]);

        let mut c = (2, 2, repeat(0.0).take(4).collect());
        Gemm::gemm(&1f32, &a, &b, &0f32, &mut c);

        assert_eq!(c.2, vec![1.0, 5.0, 1.0, 13.0]);
    }
}

pub trait Symm {
    fn symm(side: Side, symmetry: Symmetry, alpha: &Self, a: &Matrix<Self>, b: &Matrix<Self>, beta: &Self, c: &mut Matrix<Self>);
}

pub trait Hemm {
    fn hemm(side: Side, symmetry: Symmetry, alpha: &Self, a: &Matrix<Self>, b: &Matrix<Self>, beta: &Self, c: &mut Matrix<Self>);
}

macro_rules! symm_impl(($trait_name: ident, $fn_name: ident, $($t: ident), +) => (
    $(
        impl $trait_name for $t {
            fn $fn_name(side: Side, symmetry: Symmetry, alpha: &$t, a: &Matrix<$t>, b: &Matrix<$t>, beta: &$t, c: &mut Matrix<$t>) {
                unsafe {
                    prefix!($t, $fn_name)(a.order(),
                        side, symmetry,
                        a.rows(), b.cols(),
                        alpha.as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        b.as_ptr().as_c_ptr(), b.lead_dim(),
                        beta.as_const(),
                        c.as_mut_ptr().as_c_ptr(), c.lead_dim());
                }
            }
        }
    )+
));

symm_impl!(Symm, symm, f32, f64, Complex32, Complex64);
symm_impl!(Hemm, hemm, Complex32, Complex64);

pub trait Trmm {
    fn trmm(side: Side, symmetry: Symmetry, diag: Diagonal, alpha: &Self, a: &Matrix<Self>, b: &mut Matrix<Self>);
}

pub trait Trsm {
    fn trsm(side: Side, symmetry: Symmetry, diag: Diagonal, alpha: &Self, a: &Matrix<Self>, b: &mut Matrix<Self>);
}

macro_rules! trmm_impl(($trait_name: ident, $fn_name: ident, $($t: ident), +) => (
    $(
        impl $trait_name for $t {
            fn $fn_name(side: Side, symmetry: Symmetry, diag: Diagonal, alpha: &$t, a: &Matrix<$t>, b: &mut Matrix<$t>) {
                unsafe {
                    prefix!($t, $fn_name)(a.order(),
                        side, symmetry, a.transpose(), diag,
                        b.rows(), b.cols(),
                        alpha.as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        b.as_mut_ptr().as_c_ptr(), b.lead_dim());
                }
            }
        }
    )+
));

trmm_impl!(Trmm, trmm, f32, f64, Complex32, Complex64);
trmm_impl!(Trsm, trsm, Complex32, Complex64);

pub trait Herk {
    fn herk(symmetry: Symmetry, alpha: &Self, a: &Matrix<Complex<Self>>, beta: &Self, c: &mut Matrix<Complex<Self>>);
}

pub trait Her2k {
    fn her2k(symmetry: Symmetry, alpha: Complex<Self>, a: &Matrix<Complex<Self>>, b: &Matrix<Complex<Self>>, beta: &Self, c: &mut Matrix<Complex<Self>>);
}

macro_rules! herk_impl(($($t: ident), +) => (
    $(
        impl Herk for $t {
            fn herk(symmetry: Symmetry, alpha: &$t, a: &Matrix<Complex<$t>>, beta: &$t, c: &mut Matrix<Complex<$t>>) {
                unsafe {
                    prefix!(Complex<$t>, herk)(a.order(),
                        symmetry, a.transpose(),
                        a.rows(), a.cols(),
                        *alpha,
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        *beta,
                        c.as_mut_ptr().as_c_ptr(), c.lead_dim());
                }
            }
        }

        impl Her2k for $t {
            fn her2k(symmetry: Symmetry, alpha: Complex<$t>, a: &Matrix<Complex<$t>>, b: &Matrix<Complex<$t>>, beta: &$t, c: &mut Matrix<Complex<$t>>) {
                unsafe {
                    prefix!(Complex<$t>, her2k)(a.order(),
                        symmetry, a.transpose(),
                        a.rows(), a.cols(),
                        alpha.as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        b.as_ptr().as_c_ptr(), b.lead_dim(),
                        *beta,
                        c.as_mut_ptr().as_c_ptr(), c.lead_dim());
                }
            }
        }
    )+
));

herk_impl!(f32, f64);

pub trait Syrk {
    fn syrk(symmetry: Symmetry, alpha: &Self, a: &Matrix<Self>, beta: &Self, c: &mut Matrix<Self>);
}

pub trait Syr2k {
    fn syr2k(symmetry: Symmetry, alpha: &Self, a: &Matrix<Self>, b: &Matrix<Self>, beta: &Self, c: &mut Matrix<Self>);
}

macro_rules! syrk_impl(($($t: ident), +) => (
    $(
        impl Syrk for $t {
            fn syrk(symmetry: Symmetry, alpha: &$t, a: &Matrix<$t>, beta: &$t, c: &mut Matrix<$t>) {
                unsafe {
                    prefix!($t, syrk)(a.order(),
                        symmetry, a.transpose(),
                        a.rows(), a.cols(),
                        alpha.as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        beta.as_const(),
                        c.as_mut_ptr().as_c_ptr(), c.lead_dim());
                }
            }
        }

        impl Syr2k for $t {
            fn syr2k(symmetry: Symmetry, alpha: &$t, a: &Matrix<$t>, b: &Matrix<$t>, beta: &$t, c: &mut Matrix<$t>) {
                unsafe {
                    prefix!($t, syr2k)(a.order(),
                        symmetry, a.transpose(),
                        a.rows(), a.cols(),
                        alpha.as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        b.as_ptr().as_c_ptr(), b.lead_dim(),
                        beta.as_const(),
                        c.as_mut_ptr().as_c_ptr(), c.lead_dim());
                }
            }
        }
    )+
));

syrk_impl!(f32, f64, Complex32, Complex64);
