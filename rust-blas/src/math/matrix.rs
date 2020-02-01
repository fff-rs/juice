// Copyright 2015 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

use crate::attribute::Transpose;
use crate::default::Default;
use crate::math::Mat;
use crate::math::Trans;
use crate::matrix::ops::*;
use crate::matrix::Matrix;
use crate::vector::ops::*;
use num_complex::{Complex32, Complex64};
use std::ops::{Add, Mul};

impl<'a, T> Add for &'a dyn Matrix<T>
where
    T: Axpy + Copy + Default,
{
    type Output = Mat<T>;

    fn add(self, b: &dyn Matrix<T>) -> Mat<T> {
        if self.cols() != b.cols() || self.rows() != b.rows() {
            panic!("Dimension mismatch")
        }

        let scale = Default::one();
        let mut result = Mat::from(self);
        Axpy::axpy_mat(&scale, b, &mut result);
        result
    }
}

impl<'a, T> Mul<T> for &'a dyn Matrix<T>
where
    T: Sized + Copy + Scal,
{
    type Output = Mat<T>;

    fn mul(self, alpha: T) -> Mat<T> {
        let mut result = Mat::from(self);
        Scal::scal_mat(&alpha, &mut result);
        result
    }
}

macro_rules! left_scale(($($t: ident), +) => (
    $(
        impl<'a> Mul<&'a dyn Matrix<$t>> for $t
        {
            type Output = Mat<$t>;

            fn mul(self, x: &dyn Matrix<$t>) -> Mat<$t> {
                let mut result = Mat::from(x);
                Scal::scal_mat(&self, &mut result);
                result
            }
        }
    )+
));

left_scale!(f32, f64, Complex32, Complex64);

impl<'a, T> Mul<&'a dyn Matrix<T>> for &'a dyn Matrix<T>
where
    T: Default + Gemm,
{
    type Output = Mat<T>;

    fn mul(self, b: &dyn Matrix<T>) -> Mat<T> {
        if self.cols() != b.rows() {
            panic!("Dimension mismatch");
        }

        let n = self.rows() as usize;
        let m = b.cols() as usize;
        let mut result = Mat::new(n, m);
        let t = Transpose::NoTrans;

        Gemm::gemm(
            &Default::one(),
            t,
            self,
            t,
            b,
            &Default::zero(),
            &mut result,
        );
        result
    }
}

impl<'a, T> Mul<&'a dyn Matrix<T>> for Trans<&'a dyn Matrix<T>>
where
    T: Default + Gemm,
{
    type Output = Mat<T>;

    fn mul(self, b: &dyn Matrix<T>) -> Mat<T> {
        let (a, at) = match self {
            Trans::T(a) => (a, Transpose::Trans),
            Trans::H(a) => (a, Transpose::ConjTrans),
        };

        if a.rows() != b.rows() {
            panic!("Dimension mismatch");
        }

        let n = a.cols() as usize;
        let m = b.cols() as usize;
        let mut result = Mat::new(n, m);
        let bt = Transpose::NoTrans;

        Gemm::gemm(&Default::one(), at, a, bt, b, &Default::zero(), &mut result);
        result
    }
}

impl<'a, T> Mul<Trans<&'a dyn Matrix<T>>> for &'a dyn Matrix<T>
where
    T: Default + Gemm,
{
    type Output = Mat<T>;

    fn mul(self, rhs: Trans<&dyn Matrix<T>>) -> Mat<T> {
        let (b, bt) = match rhs {
            Trans::T(a) => (a, Transpose::Trans),
            Trans::H(a) => (a, Transpose::ConjTrans),
        };

        if self.cols() != b.cols() {
            panic!("Dimension mismatch");
        }

        let n = self.rows() as usize;
        let m = b.rows() as usize;
        let mut result = Mat::new(n, m);
        let at = Transpose::NoTrans;

        Gemm::gemm(
            &Default::one(),
            at,
            self,
            bt,
            b,
            &Default::zero(),
            &mut result,
        );
        result
    }
}

impl<'a, T> Mul<Trans<&'a dyn Matrix<T>>> for Trans<&'a dyn Matrix<T>>
where
    T: Default + Gemm,
{
    type Output = Mat<T>;

    fn mul(self, rhs: Trans<&dyn Matrix<T>>) -> Mat<T> {
        let (a, at) = match self {
            Trans::T(a) => (a, Transpose::Trans),
            Trans::H(a) => (a, Transpose::ConjTrans),
        };

        let (b, bt) = match rhs {
            Trans::T(a) => (a, Transpose::Trans),
            Trans::H(a) => (a, Transpose::ConjTrans),
        };

        if self.rows() != b.cols() {
            panic!("Dimension mismatch");
        }

        let n = self.cols() as usize;
        let m = b.rows() as usize;
        let mut result = Mat::new(n, m);

        Gemm::gemm(&Default::one(), at, a, bt, b, &Default::zero(), &mut result);
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::math::Marker::T;
    use crate::math::Mat;
    use crate::Matrix;

    #[test]
    fn add() {
        let a = mat![1.0, 2.0; 3.0, 4.0];
        let b = mat![-1.0, 3.0; 1.0, 1.0];

        let c = {
            let ar = &a as &dyn Matrix<_>;
            let br = &b as &dyn Matrix<_>;
            ar + br
        };

        assert_eq!(c, mat![0.0, 5.0; 4.0, 5.0]);
    }

    #[test]
    fn scale() {
        let x = mat![1f32, 2f32; 3f32, 4f32];
        let xr = &x as &dyn Matrix<_>;

        let y = xr * 3.0;
        let z = 3.0 * xr;
        assert_eq!(y, mat![3f32, 6f32; 9f32, 12f32]);
        assert_eq!(z, y);
    }

    #[test]
    fn mul() {
        let a = mat![1.0, 2.0; 3.0, 4.0];
        let b = mat![-1.0, 3.0; 1.0, 1.0];

        let c = {
            let ar = &a as &dyn Matrix<_>;
            let br = &b as &dyn Matrix<_>;
            ar * br
        };

        assert_eq!(c, mat![1.0, 5.0; 1.0, 13.0]);
    }

    #[test]
    fn left_mul_trans() {
        let a = mat![1.0, 3.0; 2.0, 4.0];
        let b = mat![-1.0, 3.0; 1.0, 1.0];

        let c = {
            let ar = &a as &dyn Matrix<_>;
            let br = &b as &dyn Matrix<_>;
            (ar ^ T) * br
        };

        assert_eq!(c, mat![1.0, 5.0; 1.0, 13.0]);
    }

    #[test]
    fn right_mul_trans() {
        let a = mat![1.0, 2.0; 3.0, 4.0];
        let b = mat![-1.0, 1.0; 3.0, 1.0];

        let c = {
            let ar = &a as &dyn Matrix<_>;
            let br = &b as &dyn Matrix<_>;
            ar * (br ^ T)
        };

        assert_eq!(c, mat![1.0, 5.0; 1.0, 13.0]);
    }

    #[test]
    fn mul_trans() {
        let a = mat![1.0, 3.0; 2.0, 4.0];
        let b = mat![-1.0, 1.0; 3.0, 1.0];

        let c = {
            let ar = &a as &dyn Matrix<_>;
            let br = &b as &dyn Matrix<_>;
            (ar ^ T) * (br ^ T)
        };

        assert_eq!(c, mat![1.0, 5.0; 1.0, 13.0]);
    }
}
