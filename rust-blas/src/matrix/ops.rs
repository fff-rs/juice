// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

//! Wrappers for matrix functions.

use crate::attribute::{Diagonal, Side, Symmetry, Transpose};
use crate::matrix::ll::*;
use crate::matrix::Matrix;
use crate::pointer::CPtr;
use crate::scalar::Scalar;
use num_complex::{Complex, Complex32, Complex64};

pub trait Gemm: Sized {
    fn gemm(
        alpha: &Self,
        at: Transpose,
        a: &dyn Matrix<Self>,
        bt: Transpose,
        b: &dyn Matrix<Self>,
        beta: &Self,
        c: &mut dyn Matrix<Self>,
    );
}

macro_rules! gemm_impl(($($t: ident), +) => (
    $(
        impl Gemm for $t {
            fn gemm(alpha: &$t, at: Transpose, a: &dyn Matrix<$t>, bt: Transpose, b: &dyn Matrix<$t>, beta: &$t, c: &mut dyn Matrix<$t>) {
                unsafe {
                    let (ar, ac)  = match at {
                        Transpose::NoTrans => (a.rows(), a.cols()),
                        _ => (a.cols(), a.rows()),
                    };
                    let (br, bc)  = match bt {
                        Transpose::NoTrans => (b.rows(), b.cols()),
                        _ => (b.cols(), b.rows()),
                    };

                    let (m, k)  = (ar, ac);
                    let n = bc;

                    if br != k || c.rows() != m || c.cols() != n {
                        panic!("Wrong GEMM dimensions: [{},{}]x[{},{}] -> [{},{}]", ar, ac, br, bc, c.rows(), c.cols());
                    }

                    prefix!($t, gemm)(a.order(),
                        at, bt,
                        m, n, k,
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
    use crate::attribute::Transpose;
    use crate::matrix::ops::Gemm;
    use crate::matrix::tests::M;
    use std::iter::repeat;

    #[test]
    fn real() {
        let a = M(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = M(2, 2, vec![-1.0, 3.0, 1.0, 1.0]);
        let t = Transpose::NoTrans;

        let mut c = M(2, 2, repeat(0.0).take(4).collect());
        Gemm::gemm(&1f32, t, &a, t, &b, &0f32, &mut c);

        assert_eq!(c.2, vec![1.0, 5.0, 1.0, 13.0]);
    }

    #[test]
    fn transpose() {
        let a = M(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = M(2, 3, vec![-1.0, 3.0, 1.0, 1.0, 1.0, 1.0]);
        let t = Transpose::Trans;

        let mut c = M(2, 2, repeat(0.0).take(4).collect());
        Gemm::gemm(&1f32, t, &a, t, &b, &0f32, &mut c);

        assert_eq!(c.2, vec![13.0, 9.0, 16.0, 12.0]);
    }
}

pub trait Symm: Sized {
    fn symm(
        side: Side,
        symmetry: Symmetry,
        alpha: &Self,
        a: &dyn Matrix<Self>,
        b: &dyn Matrix<Self>,
        beta: &Self,
        c: &mut dyn Matrix<Self>,
    );
}

pub trait Hemm: Sized {
    fn hemm(
        side: Side,
        symmetry: Symmetry,
        alpha: &Self,
        a: &dyn Matrix<Self>,
        b: &dyn Matrix<Self>,
        beta: &Self,
        c: &mut dyn Matrix<Self>,
    );
}

macro_rules! symm_impl(($trait_name: ident, $fn_name: ident, $($t: ident), +) => (
    $(
        impl $trait_name for $t {
            fn $fn_name(side: Side, symmetry: Symmetry, alpha: &$t, a: &dyn Matrix<$t>, b: &dyn Matrix<$t>, beta: &$t, c: &mut dyn Matrix<$t>) {
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

pub trait Trmm: Sized {
    fn trmm(
        side: Side,
        symmetry: Symmetry,
        trans: Transpose,
        diag: Diagonal,
        alpha: &Self,
        a: &dyn Matrix<Self>,
        b: &mut dyn Matrix<Self>,
    );
}

pub trait Trsm: Sized {
    fn trsm(
        side: Side,
        symmetry: Symmetry,
        trans: Transpose,
        diag: Diagonal,
        alpha: &Self,
        a: &dyn Matrix<Self>,
        b: &mut dyn Matrix<Self>,
    );
}

macro_rules! trmm_impl(($trait_name: ident, $fn_name: ident, $($t: ident), +) => (
    $(
        impl $trait_name for $t {
            fn $fn_name(side: Side, symmetry: Symmetry, trans: Transpose, diag: Diagonal, alpha: &$t, a: &dyn Matrix<$t>, b: &mut dyn Matrix<$t>) {
                unsafe {
                    prefix!($t, $fn_name)(a.order(),
                        side, symmetry, trans, diag,
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

pub trait Herk: Sized {
    fn herk(
        symmetry: Symmetry,
        trans: Transpose,
        alpha: &Self,
        a: &dyn Matrix<Complex<Self>>,
        beta: &Self,
        c: &mut dyn Matrix<Complex<Self>>,
    );
}

pub trait Her2k: Sized {
    fn her2k(
        symmetry: Symmetry,
        trans: Transpose,
        alpha: Complex<Self>,
        a: &dyn Matrix<Complex<Self>>,
        b: &dyn Matrix<Complex<Self>>,
        beta: &Self,
        c: &mut dyn Matrix<Complex<Self>>,
    );
}

macro_rules! herk_impl(($($t: ident), +) => (
    $(
        impl Herk for $t {
            fn herk(symmetry: Symmetry, trans: Transpose, alpha: &$t, a: &dyn Matrix<Complex<$t>>, beta: &$t, c: &mut dyn Matrix<Complex<$t>>) {
                unsafe {
                    prefix!(Complex<$t>, herk)(a.order(),
                        symmetry, trans,
                        a.rows(), a.cols(),
                        *alpha,
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        *beta,
                        c.as_mut_ptr().as_c_ptr(), c.lead_dim());
                }
            }
        }

        impl Her2k for $t {
            fn her2k(symmetry: Symmetry, trans: Transpose, alpha: Complex<$t>, a: &dyn Matrix<Complex<$t>>, b: &dyn Matrix<Complex<$t>>, beta: &$t, c: &mut dyn Matrix<Complex<$t>>) {
                unsafe {
                    prefix!(Complex<$t>, her2k)(a.order(),
                        symmetry, trans,
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

pub trait Syrk: Sized {
    fn syrk(
        symmetry: Symmetry,
        trans: Transpose,
        alpha: &Self,
        a: &dyn Matrix<Self>,
        beta: &Self,
        c: &mut dyn Matrix<Self>,
    );
}

pub trait Syr2k: Sized {
    fn syr2k(
        symmetry: Symmetry,
        trans: Transpose,
        alpha: &Self,
        a: &dyn Matrix<Self>,
        b: &dyn Matrix<Self>,
        beta: &Self,
        c: &mut dyn Matrix<Self>,
    );
}

macro_rules! syrk_impl(($($t: ident), +) => (
    $(
        impl Syrk for $t {
            fn syrk(symmetry: Symmetry, trans: Transpose, alpha: &$t, a: &dyn Matrix<$t>, beta: &$t, c: &mut dyn Matrix<$t>) {
                unsafe {
                    prefix!($t, syrk)(a.order(),
                        symmetry, trans,
                        a.rows(), a.cols(),
                        alpha.as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        beta.as_const(),
                        c.as_mut_ptr().as_c_ptr(), c.lead_dim());
                }
            }
        }

        impl Syr2k for $t {
            fn syr2k(symmetry: Symmetry, trans: Transpose, alpha: &$t, a: &dyn Matrix<$t>, b: &dyn Matrix<$t>, beta: &$t, c: &mut dyn Matrix<$t>) {
                unsafe {
                    prefix!($t, syr2k)(a.order(),
                        symmetry, trans,
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
