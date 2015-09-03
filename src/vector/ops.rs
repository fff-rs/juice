// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

//! Wrappers for vector functions.

use num::complex::{Complex, Complex32, Complex64};
use std::cmp;
use default::Default;
use pointer::CPtr;
use scalar::Scalar;
use matrix::Matrix;
use vector::ll::*;
use vector::Vector;

pub trait Copy: Sized {
    /// Copies `src.len()` elements of `src` into `dst`.
    fn copy<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(src: &V, dst: &mut W);
    /// Copies the entire matrix `dst` into `src`.
    fn copy_mat(src: &Matrix<Self>, dst: &mut Matrix<Self>);
}

macro_rules! copy_impl(($($t: ident), +) => (
    $(
        impl Copy for $t {
            fn copy<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(src: &V, dst: &mut W) {
                unsafe {
                    prefix!($t, copy)(dst.len(),
                        src.as_ptr().as_c_ptr(),  src.inc(),
                        dst.as_mut_ptr().as_c_ptr(), dst.inc());
                }
            }

            fn copy_mat(src: &Matrix<Self>, dst: &mut Matrix<Self>) {
                let len = dst.rows() * dst.cols();

                unsafe {
                    prefix!($t, copy)(len,
                        src.as_ptr().as_c_ptr(),  1,
                        dst.as_mut_ptr().as_c_ptr(), 1);
                }
            }
        }
    )+
));

copy_impl!(f32, f64, Complex32, Complex64);

/// Computes `a * x + y` and stores the result in `y`.
pub trait Axpy: Sized {
    fn axpy<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(alpha: &Self, x: &V, y: &mut W);
    fn axpy_mat(alpha: &Self, x: &Matrix<Self>, y: &mut Matrix<Self>);
}

macro_rules! axpy_impl(($($t: ident), +) => (
    $(
        impl Axpy for $t {
            fn axpy<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(alpha: &$t, x: &V, y: &mut W) {
                unsafe {
                    let n = cmp::min(x.len(), y.len());

                    prefix!($t, axpy)(n,
                        alpha.as_const(),
                        x.as_ptr().as_c_ptr(), x.inc(),
                        y.as_mut_ptr().as_c_ptr(), y.inc());
                }
            }

            fn axpy_mat(alpha: &$t, x: &Matrix<$t>, y: &mut Matrix<$t>) {
                unsafe {
                    let x_len = x.rows() * x.cols();
                    let y_len = y.rows() * y.cols();
                    let n = cmp::min(x_len, y_len);

                    prefix!($t, axpy)(n,
                        alpha.as_const(),
                        x.as_ptr().as_c_ptr(), 1,
                        y.as_mut_ptr().as_c_ptr(), 1);
                }
            }
        }
    )+
));

axpy_impl!(f32, f64, Complex32, Complex64);

#[cfg(test)]
mod axpy_tests {
    use num::complex::Complex;
    use vector::ops::Axpy;

    #[test]
    fn real() {
        let x = vec![1f32,-2f32,3f32,4f32];
        let y = vec![3f32,7f32,-2f32,2f32];
        let mut z = y.clone();

        Axpy::axpy(&1f32, &y, &mut z);
        Axpy::axpy(&1f32, &x, &mut z);
        assert_eq!(z, vec![7f32,12f32,-1f32,8f32]);
    }

    #[test]
    fn slice() {
        let x = vec![1f32,-2f32,3f32,4f32,5f32];
        let y = vec![3f32,7f32,-2f32,2f32];
        let mut z = y.clone();

        Axpy::axpy(&1f32, &y, &mut z);
        Axpy::axpy(&1f32, &x[..4], &mut z);
        assert_eq!(z, vec![7f32,12f32,-1f32,8f32]);
    }

    #[test]
    fn complex() {
        let x = vec![Complex::new(1f32, 1f32), Complex::new(1f32, 3f32)];
        let y = vec![Complex::new(3f32, -2f32), Complex::new(2f32, 3f32)];
        let mut z = x.clone();

        Axpy::axpy(&Complex::new(-1f32, 1f32), &y, &mut z);
        assert_eq!(z, vec![Complex::new(0f32, 6f32), Complex::new(-4f32, 2f32)]);
    }

}

/// Computes `a * x` and stores the result in `x`.
pub trait Scal: Sized {
    fn scal<V: ?Sized + Vector<Self>>(alpha: &Self, x: &mut V);
    fn scal_mat(alpha: &Self, x: &mut Matrix<Self>);
}

macro_rules! scal_impl(($($t: ident), +) => (
    $(
        impl Scal for $t {
            #[inline]
            fn scal<V: ?Sized + Vector<Self>>(alpha: &$t, x: &mut V) {
                unsafe {
                    prefix!($t, scal)(x.len(),
                        alpha.as_const(),
                        x.as_mut_ptr().as_c_ptr(), x.inc());
                }
            }

            fn scal_mat(alpha: &$t, x: &mut Matrix<$t>) {
                unsafe {
                    prefix!($t, scal)(x.rows() * x.cols(),
                        alpha.as_const(),
                        x.as_mut_ptr().as_c_ptr(), 1);
                }
            }
        }
    )+
));

scal_impl!(f32, f64, Complex32, Complex64);

#[cfg(test)]
mod scal_tests {
    use num::complex::Complex;
    use vector::ops::Scal;

    #[test]
    fn real() {
        let mut x = vec![1f32,-2f32,3f32,4f32];

        Scal::scal(&-2f32, &mut x);
        assert_eq!(x, vec![-2f32, 4f32, -6f32, -8f32]);
    }

    #[test]
    fn slice() {
        let mut x = vec![1f32,-2f32,3f32,4f32];

        Scal::scal(&-2f32, &mut x[..3]);
        assert_eq!(x, vec![-2f32, 4f32, -6f32, 4f32]);
    }

    #[test]
    fn complex() {
        let mut x = vec![Complex::new(1f32, 1f32), Complex::new(1f32, 3f32)];

        Scal::scal(&Complex::new(1f32, 1f32), &mut x);
        assert_eq!(x, vec![Complex::new(0f32, 2f32), Complex::new(-2f32, 4f32)]);
    }

    #[test]
    fn complex_real() {
        let mut x = vec![Complex::new(1f32, 1f32), Complex::new(1f32, 3f32)];

        Scal::scal(&Complex::new(2f32, 0f32), &mut x);
        assert_eq!(x, vec![Complex::new(2f32, 2f32), Complex::new(2f32, 6f32)]);
    }

}

/// Swaps the content of `x` and `y`.
pub trait Swap: Sized {
    /// If they are different lengths, the shorter length is used.
    fn swap<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(x: &mut V, y: &mut W);
}

macro_rules! swap_impl(($($t: ident), +) => (
    $(
        impl Swap for $t {
            fn swap<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(x: &mut V, y: &mut W) {
                unsafe {
                    let n = cmp::min(x.len(), y.len());

                    prefix!($t, swap)(n,
                        x.as_mut_ptr().as_c_ptr(), x.inc(),
                        y.as_mut_ptr().as_c_ptr(), y.inc());
                }
            }
        }
    )+
));

swap_impl!(f32, f64, Complex32, Complex64);

#[cfg(test)]
mod swap_tests {
    use num::complex::Complex;
    use vector::ops::Swap;

    #[test]
    fn real() {
        let mut x = vec![1f32,-2f32,3f32,4f32];
        let mut y = vec![2f32,-3f32,4f32,1f32];
        let xr = y.clone();
        let yr = x.clone();


        Swap::swap(&mut x, &mut y);
        assert_eq!(x, xr);
        assert_eq!(y, yr);
    }

    #[test]
    fn slice() {
        let mut x = [1f32,-2f32,3f32,4f32];
        let mut y = [2f32,-3f32,4f32,1f32];
        let xr = [2f32,-3f32,4f32,1f32];
        let yr = [1f32,-2f32,3f32,4f32];

        Swap::swap(&mut x[..], &mut y[..]);
        assert_eq!(x, xr);
        assert_eq!(y, yr);
    }

    #[test]
    fn complex() {
        let mut x = vec![Complex::new(2f32, -3f32)];
        let mut y = vec![Complex::new(-1f32, 4f32)];
        let xr = y.clone();
        let yr = x.clone();

        Swap::swap(&mut x, &mut y);
        assert_eq!(x, xr);
        assert_eq!(y, yr);
    }

}

/// Computes `x^T * y`.
pub trait Dot: Sized {
    fn dot<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(x: &V, y: &W) -> Self;
}

macro_rules! real_dot_impl(($($t: ident), +) => (
    $(
        impl Dot for $t {
            fn dot<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(x: &V, y: &W) -> $t {
                unsafe {
                    let n = cmp::min(x.len(), y.len());

                    prefix!($t, dot)(n,
                        x.as_ptr().as_c_ptr(), x.inc(),
                        y.as_ptr().as_c_ptr(), y.inc())
                }
            }
        }
    )+
));

macro_rules! complex_dot_impl(($($t: ident), +) => (
    $(
        impl Dot for $t {
            fn dot<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(x: &V, y: &W) -> $t {
                let result: $t = Default::zero();

                unsafe {
                    let n = cmp::min(x.len(), y.len());

                    prefix!($t, dotu_sub)(n,
                        x.as_ptr().as_c_ptr(), x.inc(),
                        y.as_ptr().as_c_ptr(), y.inc(),
                        (&result).as_mut());
                }

                result
            }
        }
    )+
));

real_dot_impl!(f32, f64);
complex_dot_impl!(Complex32, Complex64);

#[cfg(test)]
mod dot_tests {
    use num::complex::Complex;
    use vector::ops::Dot;

    #[test]
    fn real() {
        let x = vec![1f32,-2f32,3f32,4f32];
        let y = vec![1f32,1f32,1f32,1f32];

        let xr: f32 = Dot::dot(&x, &y);
        assert_eq!(xr, 6f32);
    }

    #[test]
    fn slice() {
        let x = [1f32,-2f32,3f32,4f32];
        let y = [1f32,1f32,1f32,1f32];

        let xr: f32 = Dot::dot(&x[..], &y[..]);
        assert_eq!(xr, 6f32);
    }

    #[test]
    fn complex() {
        let x = vec![Complex::new(1f32, 1f32), Complex::new(1f32, 3f32)];
        let y = vec![Complex::new(1f32, 1f32), Complex::new(1f32, 1f32)];

        let xr: Complex<f32> = Dot::dot(&x, &y);
        assert_eq!(xr, Complex::new(-2f32, 6f32));
    }

}

/// Computes `x^H * y`.
pub trait Dotc: Sized + Dot {
    fn dotc<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(x: &V, y: &W) -> Self {
        Dot::dot(x, y)
    }
}

macro_rules! dotc_impl(($($t: ident), +) => (
    $(
        impl Dotc for $t {
            fn dotc<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(x: &V, y: &W) -> $t {
                let result: $t = Default::zero();

                unsafe {
                    let n = cmp::min(x.len(), y.len());

                    prefix!($t, dotc_sub)(n,
                        x.as_ptr().as_c_ptr(), x.inc(),
                        y.as_ptr().as_c_ptr(), y.inc(),
                        (&result).as_mut());
                }

                result
            }
        }
    )+
));

impl Dotc for f32 {}
impl Dotc for f64 {}
dotc_impl!(Complex32, Complex64);

#[cfg(test)]
mod dotc_tests {
    use num::complex::Complex;
    use vector::ops::Dotc;

    #[test]
    fn complex_conj() {
        let x = vec![Complex::new(1f32, -1f32), Complex::new(1f32, -3f32)];
        let y = vec![Complex::new(1f32, 2f32), Complex::new(1f32, 3f32)];

        let xr: Complex<f32> = Dotc::dotc(&x, &y);
        assert_eq!(xr, Complex::new(-9f32, 9f32));
    }
}

/// Computes the sum of the absolute values of elements in a vector.
pub trait Asum: Sized {
    fn asum<V: ?Sized + Vector<Self>>(x: &V) -> Self;
}

/// Computes the L2 norm (Euclidian length) of a vector.
pub trait Nrm2: Sized {
    fn nrm2<V: ?Sized + Vector<Self>>(x: &V) -> Self;
}

macro_rules! real_norm_impl(($trait_name: ident, $fn_name: ident, $($t: ident), +) => (
    $(
        impl $trait_name for $t {
            fn $fn_name<V: ?Sized + Vector<Self>>(x: &V) -> $t {
                unsafe {
                    prefix!($t, $fn_name)(x.len(),
                        x.as_ptr().as_c_ptr(), x.inc())
                }
            }
        }
    )+
));

macro_rules! complex_norm_impl(
    ($trait_name: ident, $fn_name: ident, $t: ty, $norm_fn: ident) => (
        impl $trait_name for $t {
            fn $fn_name<V: ?Sized + Vector<Self>>(x: &V) -> $t {
                let re = unsafe {
                    $norm_fn(x.len(),
                        x.as_ptr().as_c_ptr(), x.inc())
                };

                Complex { im: 0.0, re: re }
            }
        }
    );
);

real_norm_impl!(Asum, asum, f32, f64);
real_norm_impl!(Nrm2, nrm2, f32, f64);
complex_norm_impl!(Asum, asum, Complex32, cblas_scasum);
complex_norm_impl!(Asum, asum, Complex64, cblas_dzasum);
complex_norm_impl!(Nrm2, nrm2, Complex32, cblas_scnrm2);
complex_norm_impl!(Nrm2, nrm2, Complex64, cblas_dznrm2);

#[cfg(test)]
mod asum_tests {
    use num::complex::Complex;
    use vector::ops::Asum;

    #[test]
    fn real() {
        let x = vec![1f32,-2f32,3f32,4f32];

        let r: f32 = Asum::asum(&x);
        assert_eq!(r, 10f32);
    }

    #[test]
    fn slice() {
        let x = [1f32,-2f32,3f32,4f32];

        let r: f32 = Asum::asum(&x[..]);
        assert_eq!(r, 10f32);
    }

    #[test]
    fn complex() {
        let x = vec![Complex::new(3f32, 4f32)];

        let r: Complex<f32> = Asum::asum(&x);
        assert_eq!(r, Complex { im: 0.0, re: 7f32 });
    }
}

#[cfg(test)]
mod nrm2_tests {
    use num::complex::Complex;
    use vector::ops::Nrm2;

    #[test]
    fn real() {
        let x = vec![3f32,-4f32];

        let xr: f32 = Nrm2::nrm2(&x);
        assert_eq!(xr, 5f32);
    }

    #[test]
    fn slice() {
        let x = [3f32,-4f32];

        let xr: f32 = Nrm2::nrm2(&x[..]);
        assert_eq!(xr, 5f32);
    }

    #[test]
    fn complex() {
        let x = vec![Complex::new(3f32, 4f32)];

        let xr: Complex<f32> = Nrm2::nrm2(&x);
        assert_eq!(xr, Complex { im: 0.0, re: 5f32 });
    }
}

/// Finds the index of the maximum element in a vector.
pub trait Iamax: Sized {
    fn iamax<V: ?Sized + Vector<Self>>(x: &V) -> usize;
}

macro_rules! iamax_impl(
    ($t: ty, $iamax: ident) => (
        impl Iamax for $t {
            fn iamax<V: ?Sized + Vector<Self>>(x: &V) -> usize {
                unsafe {
                    $iamax(x.len(),
                        x.as_ptr().as_c_ptr(), x.inc()) as usize
                }
            }
        }
    );
);

iamax_impl!(f32,       cblas_isamax);
iamax_impl!(f64,       cblas_idamax);
iamax_impl!(Complex32, cblas_icamax);
iamax_impl!(Complex64, cblas_izamax);

#[cfg(test)]
mod iamax_tests {
    use num::complex::Complex;
    use vector::ops::Iamax;

    #[test]
    fn real() {
        let x = vec![1f32,-2f32,3f32,4f32];

        let xr = Iamax::iamax(&x);
        assert_eq!(xr, 3usize);
    }

    #[test]
    fn slice() {
        let x = [1f32,-2f32,3f32,4f32];

        let xr = Iamax::iamax(&x[..]);
        assert_eq!(xr, 3usize);
    }

    #[test]
    fn complex() {
        let x = vec![Complex::new(3f32, 4f32), Complex::new(3f32, 5f32)];

        let xr = Iamax::iamax(&x);
        assert_eq!(xr, 1usize);
    }
}

/// Applies a Givens rotation matrix to a pair of vectors, where `cos` is
/// the value of the cosine of the angle in the Givens matrix, and `sin` is
/// the sine.
pub trait Rot: Sized {
    fn rot<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(x: &mut V, y: &mut W, cos: &Self, sin: &Self);
}

macro_rules! rot_impl(($($t: ident), +) => (
    $(
        impl Rot for $t {
            fn rot<V: ?Sized + Vector<Self>, W: ?Sized + Vector<Self>>(x: &mut V, y: &mut W, cos: &$t, sin: &$t) {
                unsafe {
                    prefix!($t, rot)(cmp::min(x.len(), y.len()),
                        x.as_mut_ptr().as_c_ptr(), x.inc(),
                        y.as_mut_ptr().as_c_ptr(), y.inc(),
                        cos.as_const(), sin.as_const());
                }
            }
        }
    )+
));

rot_impl!(f32, f64);

#[cfg(test)]
mod rot_tests {
    use vector::ops::{
        Scal,
        Rot,
    };

    #[test]
    fn real() {
        let mut x = vec![1f32,-2f32,3f32,4f32];
        let mut y = vec![3f32,7f32,-2f32,2f32];
        let cos = 0f32;
        let sin = 1f32;

        let xr = y.clone();
        let mut yr = x.clone();
        Scal::scal(&-1f32, &mut yr);

        Rot::rot(&mut x, &mut y, &cos, &sin);
        assert_eq!(x, xr);
        assert_eq!(y, yr);
    }
}
