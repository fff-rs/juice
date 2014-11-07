// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

use attribute::{
    Order,
    RowMajor,
    Transpose,
    NoTrans,
};

pub mod ll;

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
