// Copyright 2015 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

use std::ops::{
    Mul,
};
use attribute::Transpose;
use default::Default;
use matrix::ops::*;
use matrix::Matrix;
use math::Mat;

impl<'a, T> Mul<&'a Matrix<T>> for &'a Matrix<T>
    where T: Default + Gemm,
{
    type Output = Mat<T>;

    fn mul(self, b: &'a Matrix<T>) -> Mat<T> {
        if self.cols() != b.rows() {
            panic!("Dimension mismatch");
        }

        let n = self.rows() as usize;
        let m = b.cols() as usize;
        let mut result = Mat::new(n, m);
        let t = Transpose::NoTrans;

        Gemm::gemm(&Default::one(), t, self, t, b, &Default::zero(), &mut result);
        result
    }
}

#[cfg(test)]
mod tests {
    use Matrix;
    use math::Mat;

    #[test]
    fn add() {
        let a = mat![1.0, 2.0; 3.0, 4.0];
        let b = mat![-1.0, 3.0; 1.0, 1.0];

        let c = {
            let ar = &a as &Matrix<_>;
            let br = &b as &Matrix<_>;
            ar + br
        };

        assert_eq!(c, mat![0.0, 5.0; 4.0, 5.0]);
    }

    #[test]
    fn scale() {
        let x = mat![1f32, 2f32; 3f32, 4f32];
        let xr = &x as &Matrix<_>;

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
            let ar = &a as &Matrix<_>;
            let br = &b as &Matrix<_>;
            ar * br
        };

        assert_eq!(c, mat![1.0, 5.0; 1.0, 13.0]);
    }
}
