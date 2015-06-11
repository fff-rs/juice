// Copyright 2015 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

use std::ops::{
    Add,
};
use default::Default;
use vector::ops::*;
use vector::Vector;

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

#[cfg(test)]
mod tests {
    use Vector;

    #[test]
    fn add() {
        let x = vec![1f32, 2f32];
        let y = vec![3f32, 4f32];

        let z = &x as &Vector<_> + &y;

        assert_eq!(&z, &vec![4f32, 6f32]);
    }
}
