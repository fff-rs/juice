// Copyright 2015 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

use crate::attribute::Transpose;
use crate::default::Default;
use crate::math::Mat;
use crate::math::Trans;
use crate::matrix::Matrix;
use crate::matrix_vector::ops::*;
use crate::vector::Vector;
use std::ops::Mul;

impl<'a, T> Mul<&'a dyn Vector<T>> for &'a dyn Matrix<T>
where
    T: Default + Copy + Gemv,
{
    type Output = Vec<T>;

    fn mul(self, x: &dyn Vector<T>) -> Vec<T> {
        let n = self.rows() as usize;
        let mut result = Vec::with_capacity(n);
        unsafe {
            result.set_len(n);
        }
        let scale = Default::one();
        let clear = Default::zero();
        let t = Transpose::NoTrans;

        Gemv::gemv(t, &scale, self, x, &clear, &mut result);
        result
    }
}

impl<'a, T> Mul<Trans<&'a dyn Vector<T>>> for &'a dyn Vector<T>
where
    T: Default + Ger + Gerc + Clone,
{
    type Output = Mat<T>;

    fn mul(self, x: Trans<&dyn Vector<T>>) -> Mat<T> {
        let n = self.len() as usize;
        let m = (*x).len() as usize;
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
    use crate::math::Marker::T;
    use crate::math::Mat;
    use crate::Matrix;
    use crate::Vector;

    #[test]
    fn mul() {
        let a = mat![2f32, -2.0; 2.0, -4.0];
        let x = vec![2f32, 1.0];

        let y = {
            let ar = &a as &dyn Matrix<f32>;
            let xr = &x as &dyn Vector<f32>;
            ar * xr
        };

        assert_eq!(y, vec![2.0, 0.0]);
    }

    #[test]
    fn outer() {
        let x = vec![2.0, 1.0, 4.0];
        let y = vec![3.0, 6.0, -1.0];

        let a = {
            let xr = &x as &dyn Vector<_>;
            let yr = &y as &dyn Vector<_>;

            xr * (yr ^ T)
        };

        let result = mat![  6.0, 12.0, -2.0;
                            3.0, 6.0, -1.0;
                            12.0, 24.0, -4.0];
        assert_eq!(a, result);
    }
}
