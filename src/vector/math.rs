// Copyright 2015 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

use std::ops::{
    Add,
    BitXor,
    Mul,
};
use default::Default;
use vector::ops::*;
use vector::Vector;
use transpose::{
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

#[cfg(test)]
mod tests {
    use Vector;
    use transpose::Marker::T;

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
}
