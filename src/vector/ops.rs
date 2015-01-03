// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

use num::complex::{Complex, Complex32, Complex64};
use std::cmp;
use default::Default;
use pointer::CPtr;
use scalar::Scalar;
use vector;
use vector::Vector;

pub trait Copy {
    fn copy(x: &Vector<Self>, y: &mut Vector<Self>);
}

macro_rules! copy_impl(
    ($t: ty, $copy_fn: ident) => (
        impl Copy for $t {
            fn copy(x: &Vector<$t>, y: &mut Vector<$t>) {
                unsafe {
                    vector::ll::$copy_fn(x.len(),
                        x.as_ptr().as_c_ptr(),  x.inc(),
                        y.as_mut_ptr().as_c_ptr(), y.inc());
                }
            }
        }
    );
);

copy_impl!(f32,       cblas_scopy);
copy_impl!(f64,       cblas_dcopy);
copy_impl!(Complex32, cblas_ccopy);
copy_impl!(Complex64, cblas_zcopy);

pub trait Axpy {
    fn axpy(alpha: &Self, x: &Vector<Self>, y: &mut Vector<Self>);
}

macro_rules! axpy_impl(
    ($t: ty, $update_fn: ident) => (
        impl Axpy for $t {
            fn axpy(alpha: &$t, x: &Vector<$t>, y: &mut Vector<$t>) {
                unsafe {
                    let n = cmp::min(x.len(), y.len());

                    vector::ll::$update_fn(n,
                        alpha.as_const(),
                        x.as_ptr().as_c_ptr(), x.inc(),
                        y.as_mut_ptr().as_c_ptr(), y.inc());
                }
            }
        }
    );
);

axpy_impl!(f32,       cblas_saxpy);
axpy_impl!(f64,       cblas_daxpy);
axpy_impl!(Complex32, cblas_caxpy);
axpy_impl!(Complex64, cblas_zaxpy);

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
    fn complex() {
        let x = vec![Complex::new(1f32, 1f32), Complex::new(1f32, 3f32)];
        let y = vec![Complex::new(3f32, -2f32), Complex::new(2f32, 3f32)];
        let mut z = x.clone();

        Axpy::axpy(&Complex::new(-1f32, 1f32), &y, &mut z);
        assert_eq!(z, vec![Complex::new(0f32, 6f32), Complex::new(-4f32, 2f32)]);
    }

}

pub trait Scal {
    fn scal(alpha: &Self, x: &mut Vector<Self>);
}

macro_rules! scal_impl(
    ($t: ty, $scal_fn: ident) => (
        impl Scal for $t {
            #[inline]
            fn scal(alpha: &$t, x: &mut Vector<$t>) {
                unsafe {
                    vector::ll::$scal_fn(x.len(),
                        *alpha,
                        x.as_mut_ptr().as_c_ptr(), x.inc());
                }
            }
        }
    );

    ($t: ty, $scal_fn: ident, $real_scal_fn: ident) => (
        impl Scal for $t {
            #[inline]
            fn scal(alpha: &$t, x: &mut Vector<$t>) {
                if alpha.im == 0.0 {
                    unsafe {
                        vector::ll::$real_scal_fn(x.len(),
                            alpha.re,
                            x.as_mut_ptr().as_c_ptr(), x.inc());
                    }
                } else {
                    unsafe {
                        vector::ll::$scal_fn(x.len(),
                            alpha.as_const(),
                            x.as_mut_ptr().as_c_ptr(), x.inc());
                    }
                }
            }
        }
    );
);

scal_impl!(f32, cblas_sscal);
scal_impl!(f64, cblas_dscal);
scal_impl!(Complex32, cblas_cscal, cblas_csscal);
scal_impl!(Complex64, cblas_zscal, cblas_zdscal);

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

pub trait Swap {
    fn swap(x: &mut Vector<Self>, y: &mut Vector<Self>);
}

macro_rules! swap_impl(
    ($t: ty, $swap_fn: ident) => (
        impl Swap for $t {
            fn swap(x: &mut Vector<$t>, y: &mut Vector<$t>) {
                unsafe {
                    let n = cmp::min(x.len(), y.len());

                    vector::ll::$swap_fn(n,
                        x.as_mut_ptr().as_c_ptr(), x.inc(),
                        y.as_mut_ptr().as_c_ptr(), y.inc());
                }
            }
        }
    );
);

swap_impl!(f32,       cblas_sswap);
swap_impl!(f64,       cblas_dswap);
swap_impl!(Complex32, cblas_cswap);
swap_impl!(Complex64, cblas_zswap);

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

pub trait Dot {
    fn dot(x: &Vector<Self>, y: &Vector<Self>) -> Self;
}

macro_rules! real_dot_impl(
    ($t: ty, $dot_fn: ident) => (
        impl Dot for $t {
            fn dot(x: &Vector<$t>, y: &Vector<$t>) -> $t {
                unsafe {
                    let n = cmp::min(x.len(), y.len());

                    vector::ll::$dot_fn(n,
                        x.as_ptr().as_c_ptr(), x.inc(),
                        y.as_ptr().as_c_ptr(), y.inc()) as $t
                }
            }
        }
    );
);

macro_rules! complex_dot_impl(
    ($t: ty, $dot_fn: ident) => (
        impl Dot for $t {
            fn dot(x: &Vector<$t>, y: &Vector<$t>) -> $t {
                let result: $t = Default::zero();

                unsafe {
                    let n = cmp::min(x.len(), y.len());

                    vector::ll::$dot_fn(n,
                        x.as_ptr().as_c_ptr(), x.inc(),
                        y.as_ptr().as_c_ptr(), y.inc(),
                        (&result).as_mut());
                }

                result
            }
        }
    );
);

real_dot_impl!(f32, cblas_sdot);
real_dot_impl!(f64, cblas_ddot);
complex_dot_impl!(Complex32, cblas_cdotu_sub);
complex_dot_impl!(Complex64, cblas_zdotu_sub);

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
    fn complex() {
        let x = vec![Complex::new(1f32, 1f32), Complex::new(1f32, 3f32)];
        let y = vec![Complex::new(1f32, 1f32), Complex::new(1f32, 1f32)];

        let xr: Complex<f32> = Dot::dot(&x, &y);
        assert_eq!(xr, Complex::new(-2f32, 6f32));
    }

}

pub trait Dotc: Dot {
    fn dotc(x: &Vector<Self>, y: &Vector<Self>) -> Self {
        Dot::dot(x, y)
    }
}

macro_rules! dot_impl(
    ($t: ty, $dotc_fn: ident) => (
        impl Dotc for $t {
            fn dotc(x: &Vector<$t>, y: &Vector<$t>) -> $t {
                let result: $t = Default::zero();

                unsafe {
                    let n = cmp::min(x.len(), y.len());

                    vector::ll::$dotc_fn(n,
                        x.as_ptr().as_c_ptr(), x.inc(),
                        y.as_ptr().as_c_ptr(), y.inc(),
                        (&result).as_mut());
                }

                result
            }
        }
    );
);

impl Dotc for f32 {}
impl Dotc for f64 {}
dot_impl!(Complex32, cblas_cdotc_sub);
dot_impl!(Complex64, cblas_zdotc_sub);

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

pub trait Asum {
    fn asum(x: &Vector<Self>) -> Self;
}

pub trait Nrm2 {
    fn nrm2(x: &Vector<Self>) -> Self;
}

macro_rules! real_norm_impl(
    ($trait_name: ident, $fn_name: ident, $t: ty, $norm_fn: ident) => (
        impl $trait_name for $t {
            fn $fn_name(x: &Vector<$t>) -> $t {
                unsafe {
                    vector::ll::$norm_fn(x.len(),
                        x.as_ptr().as_c_ptr(), x.inc())
                }
            }
        }
    );
);

macro_rules! complex_norm_impl(
    ($trait_name: ident, $fn_name: ident, $t: ty, $norm_fn: ident) => (
        impl $trait_name for $t {
            fn $fn_name(x: &Vector<$t>) -> $t {
                let re = unsafe {
                    vector::ll::$norm_fn(x.len(),
                        x.as_ptr().as_c_ptr(), x.inc())
                };

                Complex { im: 0.0, re: re }
            }
        }
    );
);

real_norm_impl!(Asum, asum, f32, cblas_sasum);
real_norm_impl!(Asum, asum, f64, cblas_dasum);
complex_norm_impl!(Asum, asum, Complex32, cblas_scasum);
complex_norm_impl!(Asum, asum, Complex64, cblas_dzasum);
real_norm_impl!(Nrm2, nrm2, f32, cblas_snrm2);
real_norm_impl!(Nrm2, nrm2, f64, cblas_dnrm2);
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
    fn complex() {
        let x = vec![Complex::new(3f32, 4f32)];

        let xr: Complex<f32> = Nrm2::nrm2(&x);
        assert_eq!(xr, Complex { im: 0.0, re: 5f32 });
    }
}

pub trait Iamax {
    fn iamax(x: &Vector<Self>) -> uint;
}

macro_rules! iamax_impl(
    ($t: ty, $iamax: ident) => (
        impl Iamax for $t {
            fn iamax(x: &Vector<$t>) -> uint {
                unsafe {
                    vector::ll::$iamax(x.len(),
                        x.as_ptr().as_c_ptr(), x.inc()) as uint
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
        assert_eq!(xr, 3u);
    }

    #[test]
    fn complex() {
        let x = vec![Complex::new(3f32, 4f32), Complex::new(3f32, 5f32)];

        let xr = Iamax::iamax(&x);
        assert_eq!(xr, 1u);
    }
}


pub trait Rot {
    fn rot(x: &mut Vector<Self>, y: &mut Vector<Self>, cos: &Self, sin: &Self);
}

macro_rules! rot_impl(
    ($t: ty, $rot_fn: ident) => (
        impl Rot for $t {
            fn rot(x: &mut Vector<$t>, y: &mut Vector<$t>, cos: &$t, sin: &$t) {
                unsafe {
                    vector::ll::$rot_fn(cmp::min(x.len(), y.len()),
                        x.as_mut_ptr().as_c_ptr(), x.inc(),
                        y.as_mut_ptr().as_c_ptr(), y.inc(),
                        cos.as_const(), sin.as_const());
                }
            }
        }
    );
);

rot_impl!(f32, cblas_srot);
rot_impl!(f64, cblas_drot);

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
