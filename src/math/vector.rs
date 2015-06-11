// Copyright 2015 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

use std::ops::{
    Add,
    BitXor,
    Mul,
};
use num::complex::{Complex32, Complex64};
use default::Default;
use vector::ops::*;
use vector::Vector;
use math::{
    Marker,
    Trans,
};

impl<'a, T> BitXor<Marker> for &'a Vector<T>
{
    type Output = Trans<&'a Vector<T>>;

    fn bitxor(self, m: Marker) -> Trans<&'a Vector<T>> {
        match m {
            Marker::T => Trans::T(self),
            Marker::H => Trans::H(self),
        }
    }
}

impl<'a, T> Add<&'a Vector<T>> for &'a Vector<T>
    where T: Axpy + Copy + Default
{
    type Output = Vec<T>;

    fn add(self, rhs: &Vector<T>) -> Vec<T> {
        let mut v: Vec<T> = self.into();
        let scale = Default::one();

        Axpy::axpy(&scale, rhs, &mut v);
        Vec::from(v)
    }
}

impl<'a, T, V> Mul<&'a V> for Trans<&'a Vector<T>>
    where T: Sized + Copy + Dot + Dotc,
          V: Vector<T>,
{
    type Output = T;

    fn mul(self, x: &'a V) -> T {
        match self {
            Trans::T(v) => Dot::dot(v, x),
            Trans::H(v) => Dotc::dotc(v, x),
        }
    }
}

impl<'a, T> Mul<T> for &'a Vector<T>
    where T: Sized + Copy + Scal
{
    type Output = Vec<T>;

    fn mul(self, alpha: T) -> Vec<T> {
        let mut result: Vec<_> = self.into();
        Scal::scal(&alpha, &mut result);
        result
    }
}

macro_rules! left_scale(($($t: ident), +) => (
    $(
        impl<'a> Mul<&'a Vector<$t>> for $t
        {
            type Output = Vec<$t>;

            fn mul(self, x: &'a Vector<$t>) -> Vec<$t> {
                let mut result: Vec<_> = x.into();
                Scal::scal(&self, &mut result);
                result
            }
        }
    )+
));

left_scale!(f32, f64, Complex32, Complex64);

#[cfg(test)]
mod tests {
    use Vector;
    use math::Marker::{T, H};
    use num::complex::Complex;

    #[test]
    fn add() {
        let x = vec![1f32, 2f32];
        let y = vec![3f32, 4f32];

        let z = &x as &Vector<_> + &y;

        assert_eq!(&z, &vec![4f32, 6f32]);
    }

    #[test]
    fn dot() {
        let x = vec![1f32, 2f32];
        let y = vec![-1f32, 2f32];

        let dot = {
            let z = &x as &Vector<_>;
            (z ^ T) * &y
        };

        assert_eq!(dot, 3.0);
    }

    #[test]
    fn herm_dot() {
        let x = vec![Complex::new(1f32, -1f32), Complex::new(1f32, -3f32)];
        let y = vec![Complex::new(1f32, 2f32), Complex::new(1f32, 3f32)];

        let dot = {
            let z = &x as &Vector<_>;
            (z ^ H) * &y
        };

        assert_eq!(dot, Complex::new(-9f32, 9f32));
    }

    #[test]
    fn scale() {
        let x = vec![1f32, 2f32];
        let xr = &x as &Vector<_>;

        let y = xr * 3.0;
        let z = 3.0 * xr;
        assert_eq!(y, vec![3f32, 6f32]);
        assert_eq!(z, y);
    }
}
