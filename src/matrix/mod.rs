// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

use attribute::{
    Order,
    Transpose,
};

pub mod ll;
pub mod ops;

pub trait Matrix<T> {
    fn lead_dim(&self) -> i32 { self.rows() }
    fn order(&self) -> Order { Order::RowMajor }
    fn transpose(&self) -> Transpose { Transpose::NoTrans }
    fn rows(&self) -> i32;
    fn cols(&self) -> i32;
    fn as_ptr(&self) -> *const T;
    fn as_mut_ptr(&mut self) -> *mut T;
}

pub trait BandMatrix<T>: Matrix<T> {
    fn sub_diagonals(&self) -> i32;
    fn sup_diagonals(&self) -> i32;
}

#[cfg(test)]
mod test_struct {
    use matrix::Matrix;

    impl<T> Matrix<T> for (i32, i32, Vec<T>) {
        fn rows(&self) -> i32 {
            self.0
        }

        fn cols(&self) -> i32 {
            self.1
        }

        #[inline]
        fn as_ptr(&self) -> *const T {
            self.2[..].as_ptr()
        }

        #[inline]
        fn as_mut_ptr(&mut self) -> *mut T {
            (&mut self.2[..]).as_mut_ptr()
        }
    }
}
