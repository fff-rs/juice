// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

use attribute::{
    Order,
    RowMajor,
    Transpose,
    NoTrans,
};
use mat::Mat;

pub mod ll;
pub mod ops;

pub trait BlasMatrix<T> {
    fn lead_dim(&self) -> i32 { self.rows() }
    fn order(&self) -> Order { RowMajor }
    fn transpose(&self) -> Transpose { NoTrans }
    fn rows(&self) -> i32;
    fn cols(&self) -> i32;
    fn as_ptr(&self) -> *const T;
    fn as_mut_ptr(&mut self) -> *mut T;
}

pub trait BandMatrix<T>: BlasMatrix<T> {
    fn sub_diagonals(&self) -> i32;
    fn sup_diagonals(&self) -> i32;
}

impl<T> BlasMatrix<T> for Mat<T> {
    #[inline]
    fn rows(&self) -> i32 {
        let l: Option<i32> = NumCast::from(self.rows());
        match l {
            Some(l) => l,
            None => panic!(),
        }
    }

    #[inline]
    fn cols(&self) -> i32 {
        let l: Option<i32> = NumCast::from(self.cols());
        match l {
            Some(l) => l,
            None => panic!(),
        }
    }

    #[inline]
    fn as_ptr(&self) -> *const T {
        unsafe { self.as_slice().as_ptr() }
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T {
        unsafe { self.as_mut_slice().as_mut_ptr() }
    }
}
