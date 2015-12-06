// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

//! Matrix operations.

use attribute::{
    Order,
};

pub mod ll;
pub mod ops;

/// Methods that allow a type to be used in BLAS functions as a matrix.
pub trait Matrix<T> {
    /// The leading dimension of the matrix. Defaults to `cols` for `RowMajor`
    /// order and 'rows' for `ColMajor` order.
    fn lead_dim(&self) -> i32 {
        match self.order() {
            Order::RowMajor => self.cols(),
            Order::ColMajor => self.rows(),
        }
    }
    /// The order of the matrix. Defaults to `RowMajor`.
    fn order(&self) -> Order { Order::RowMajor }
    /// Returns the number of rows.
    fn rows(&self) -> i32;
    /// Returns the number of columns.
    fn cols(&self) -> i32;
    /// An unsafe pointer to a contiguous block of memory.
    fn as_ptr(&self) -> *const T;
    /// An unsafe pointer to a contiguous block of memory.
    fn as_mut_ptr(&mut self) -> *mut T;
}

pub trait BandMatrix<T>: Matrix<T> {
    fn sub_diagonals(&self) -> i32;
    fn sup_diagonals(&self) -> i32;
}

#[cfg(test)]
pub mod tests {
    use matrix::Matrix;

    pub struct M<T>(pub i32, pub i32, pub Vec<T>);

    impl<T> Matrix<T> for M<T> {
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
