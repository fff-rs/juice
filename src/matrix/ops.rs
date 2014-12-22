// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

extern crate num;

use self::num::complex::{Complex, Complex32, Complex64};
use attribute::{Diagonal, Side, Symmetry};
use pointer::CPtr;
use scalar::Scalar;
use matrix;
use matrix::Matrix;
use vector::Vector;

pub trait Gemm {
    fn gemm(alpha: &Self, a: &Matrix<Self>, b: &Matrix<Self>, beta: &Self, c: &mut Matrix<Self>);
}

macro_rules! gemm_impl(
    ($t: ty, $gemm_fn: ident) => (
        impl Gemm for $t {
            fn gemm(alpha: &$t, a: &Matrix<$t>, b: &Matrix<$t>, beta: &$t, c: &mut Matrix<$t>) {
                unsafe {
                    matrix::ll::$gemm_fn(a.order(),
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
    );
);

gemm_impl!(f32,       cblas_sgemm);
gemm_impl!(f64,       cblas_dgemm);
gemm_impl!(Complex32, cblas_cgemm);
gemm_impl!(Complex64, cblas_zgemm);

#[cfg(test)]
mod gemm_tests {
    extern crate num;
    extern crate test;

    use matrix::ops::Gemm;

    #[test]
    fn real() {
        let a = (2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = (2, 2, vec![-1.0, 3.0, 1.0, 1.0]);

        let mut c = (2, 2, Vec::from_elem(4, 0.0));
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

macro_rules! symm_impl(
    ($trait_name: ident, $fn_name: ident, $t: ty, $symm_fn: ident) => (
        impl $trait_name for $t {
            fn $fn_name(side: Side, symmetry: Symmetry, alpha: &$t, a: &Matrix<$t>, b: &Matrix<$t>, beta: &$t, c: &mut Matrix<$t>) {
                unsafe {
                    matrix::ll::$symm_fn(a.order(),
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
    );
);

symm_impl!(Symm, symm, f32,       cblas_ssymm);
symm_impl!(Symm, symm, f64,       cblas_dsymm);
symm_impl!(Symm, symm, Complex32, cblas_csymm);
symm_impl!(Symm, symm, Complex64, cblas_zsymm);

symm_impl!(Hemm, hemm, Complex32, cblas_chemm);
symm_impl!(Hemm, hemm, Complex64, cblas_zhemm);

pub trait Trmm {
    fn trmm(side: Side, symmetry: Symmetry, diag: Diagonal, alpha: &Self, a: &Matrix<Self>, b: &mut Matrix<Self>);
}

pub trait Trsm {
    fn trsm(side: Side, symmetry: Symmetry, diag: Diagonal, alpha: &Self, a: &Matrix<Self>, b: &mut Matrix<Self>);
}

macro_rules! trmm_impl(
    ($trait_name: ident, $fn_name: ident, $t: ty, $symm_fn: ident) => (
        impl $trait_name for $t {
            fn $fn_name(side: Side, symmetry: Symmetry, diag: Diagonal, alpha: &$t, a: &Matrix<$t>, b: &mut Matrix<$t>) {
                unsafe {
                    matrix::ll::$symm_fn(a.order(),
                        side, symmetry, a.transpose(), diag,
                        b.rows(), b.cols(),
                        alpha.as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        b.as_mut_ptr().as_c_ptr(), b.lead_dim());
                }
            }
        }
    );
);

trmm_impl!(Trmm, trmm, f32,       cblas_strmm);
trmm_impl!(Trmm, trmm, f64,       cblas_dtrmm);
trmm_impl!(Trmm, trmm, Complex32, cblas_ctrmm);
trmm_impl!(Trmm, trmm, Complex64, cblas_ztrmm);

trmm_impl!(Trsm, trsm, Complex32, cblas_ctrsm);
trmm_impl!(Trsm, trsm, Complex64, cblas_ztrsm);

pub trait Herk {
    fn herk(symmetry: Symmetry, alpha: &Self, a: &Matrix<Complex<Self>>, beta: &Self, c: &mut Matrix<Complex<Self>>);
}

macro_rules! herk_impl(
    ($t: ty, $symm_fn: ident) => (
        impl Herk for $t {
            fn herk(symmetry: Symmetry, alpha: &$t, a: &Matrix<Complex<$t>>, beta: &$t, c: &mut Matrix<Complex<$t>>) {
                unsafe {
                    matrix::ll::$symm_fn(a.order(),
                        symmetry, a.transpose(),
                        a.rows(), a.cols(),
                        *alpha,
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        *beta,
                        c.as_mut_ptr().as_c_ptr(), c.lead_dim());
                }
            }
        }
    );
);

herk_impl!(f32, cblas_cherk);
herk_impl!(f64, cblas_zherk);

pub trait Her2k {
    fn her2k(symmetry: Symmetry, alpha: Complex<Self>, a: &Matrix<Complex<Self>>, b: &Matrix<Complex<Self>>, beta: &Self, c: &mut Matrix<Complex<Self>>);
}

macro_rules! her2k_impl(
    ($t: ty, $her2k_fn: ident) => (
        impl Her2k for $t {
            fn her2k(symmetry: Symmetry, alpha: Complex<$t>, a: &Matrix<Complex<$t>>, b: &Matrix<Complex<$t>>, beta: &$t, c: &mut Matrix<Complex<$t>>) {
                unsafe {
                    matrix::ll::$her2k_fn(a.order(),
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
    );
);

her2k_impl!(f32, cblas_cher2k);
her2k_impl!(f64, cblas_zher2k);

pub trait Syrk {
    fn syrk(symmetry: Symmetry, alpha: &Self, a: &Matrix<Self>, beta: &Self, c: &mut Matrix<Self>);
}

macro_rules! syrk_impl(
    ($t: ty, $symm_fn: ident) => (
        impl Syrk for $t {
            fn syrk(symmetry: Symmetry, alpha: &$t, a: &Matrix<$t>, beta: &$t, c: &mut Matrix<$t>) {
                unsafe {
                    matrix::ll::$symm_fn(a.order(),
                        symmetry, a.transpose(),
                        a.rows(), a.cols(),
                        alpha.as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        beta.as_const(),
                        c.as_mut_ptr().as_c_ptr(), c.lead_dim());
                }
            }
        }
    );
);

syrk_impl!(f32,       cblas_ssyrk);
syrk_impl!(f64,       cblas_dsyrk);
syrk_impl!(Complex32, cblas_csyrk);
syrk_impl!(Complex64, cblas_zsyrk);

pub trait Syr2k {
    fn syr2k(symmetry: Symmetry, alpha: &Self, a: &Matrix<Self>, b: &Matrix<Self>, beta: &Self, c: &mut Matrix<Self>);
}

macro_rules! syr2k_impl(
    ($t: ty, $syr2k_fn: ident) => (
        impl Syr2k for $t {
            fn syr2k(symmetry: Symmetry, alpha: &$t, a: &Matrix<$t>, b: &Matrix<$t>, beta: &$t, c: &mut Matrix<$t>) {
                unsafe {
                    matrix::ll::$syr2k_fn(a.order(),
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
    );
);

syr2k_impl!(f32,       cblas_ssyr2k);
syr2k_impl!(f64,       cblas_dsyr2k);
syr2k_impl!(Complex32, cblas_csyr2k);
syr2k_impl!(Complex64, cblas_zsyr2k);
