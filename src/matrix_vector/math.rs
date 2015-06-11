// Copyright 2015 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

use std::ops::{
    Mul,
};
use default::Default;
use vector::Vector;
use matrix_vector::ops::*;
use matrix::Matrix;

impl<'a, T, V> Mul<&'a V> for &'a Matrix<T>
    where T: Default + Copy + Gemv,
          V: Vector<T>,
{
    type Output = Vec<T>;

    fn mul(self, x: &'a V) -> Vec<T> {
        let n = self.rows() as usize;
        let mut result = Vec::with_capacity(n);
        unsafe { result.set_len(n); }
        let scale = Default::one();
        let clear = Default::zero();

        Gemv::gemv(&scale, self, x, &clear, &mut result);
        result
    }
}

#[cfg(test)]
mod tests {
    use Vector;
    use Matrix;

    #[test]
    fn mul() {
        let a = (2, 2, vec![2.0, -2.0, 2.0, -4.0]);
        let x = vec![2.0, 1.0];

        let y = {
            let ar = &a as &Matrix<_>;
            ar * &x
        };

        assert_eq!(y, vec![2.0, 0.0]);
    }
}
