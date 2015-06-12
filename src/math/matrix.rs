// Copyright 2015 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

use std::ops::{
    Mul,
};
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

        Gemm::gemm(&Default::one(), self, b, &Default::zero(), &mut result);
        result
    }
}

#[cfg(test)]
mod tests {
    use Matrix;
    use math::Mat;

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
