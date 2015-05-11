// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

//! Matrix operations.

use attribute::{
    Order,
    Transpose,
};

pub mod ll;
pub mod ops;

/// Methods that allow a type to be used in BLAS functions as a matrix.
pub trait Matrix<T> {
    /// The leading dimension of the matrix. Defaults to `rows`.
    fn lead_dim(&self) -> i32 { self.rows() }
    /// The order of the matrix. Defaults to `RowMajor`.
    fn order(&self) -> Order { Order::RowMajor }
    /// Returns the currently applied transpose.
    fn transpose(&self) -> Transpose { Transpose::NoTrans }
    /// Returns the number of rows.
    fn rows(&self) -> i32;
    /// Returns the number of columns.
    fn cols(&self) -> i32;
    /// An unsafe pointer to a contiguous block of memory.
    unsafe fn as_ptr(&self) -> *const T;
    /// An unsafe pointer to a contiguous block of memory.
    unsafe fn as_mut_ptr(&mut self) -> *mut T;
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
        unsafe fn as_ptr(&self) -> *const T {
            self.2[..].as_ptr()
        }

        #[inline]
        unsafe fn as_mut_ptr(&mut self) -> *mut T {
            (&mut self.2[..]).as_mut_ptr()
        }
    }
}
