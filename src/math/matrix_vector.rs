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
use math::Trans;
use math::Mat;

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

impl<'a, T, V> Mul<Trans<&'a V>> for &'a Vector<T>
    where T: Default + Copy + Ger + Gerc + Clone,
          V: Vector<T>,
{
    type Output = Mat<T>;

    fn mul(self, x: Trans<&'a V>) -> Mat<T> {
        let inner = match x {
            Trans::T(v) => v,
            Trans::H(v) => v,
        };

        let n = self.len() as usize;
        let m = inner.len() as usize;
        let mut result = Mat::fill(Default::zero(), n, m);
        let scale = Default::one();

        match x {
            Trans::T(v) => Ger::ger(&scale, self, v, &mut result),
            Trans::H(v) => Gerc::gerc(&scale, self, v, &mut result),
        }

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
