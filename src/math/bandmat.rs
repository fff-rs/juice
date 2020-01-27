// Copyright 2015 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.
use libc::c_int as c_int;
use num::traits::NumCast;
use std::fmt;
use std::iter::repeat;
use std::ops::Index;
use std::slice;
use vector::ops::Copy;
use matrix::BandMatrix;
use Matrix;
use math::Mat;

#[derive(Debug, PartialEq)]
pub struct BandMat<T> {
    rows: usize,
    cols: usize,
    sub_diagonals: c_int,
    sup_diagonals: c_int,
    data: Vec<T>,
}

impl<T> BandMat<T> {
    pub fn new(n: usize, m: usize, sub: c_int, sup: c_int) -> BandMat<T> {
        let len = n * m;
        let mut data = Vec::with_capacity(len);
        unsafe {
            data.set_len(len);
        }

        BandMat {
            rows: n,
            cols: m,
            data: data,
            sub_diagonals: sub,
            sup_diagonals: sup,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub unsafe fn set_rows(&mut self, n: usize) {
        self.rows = n;
    }
    pub unsafe fn set_cols(&mut self, n: usize) {
        self.cols = n;
    }

    pub unsafe fn set_sub_diagonals(&mut self, n: c_int) {
        self.sub_diagonals = n;
    }
    pub unsafe fn set_sup_diagonals(&mut self, n: c_int) {
        self.sup_diagonals = n;
    }

    pub unsafe fn push(&mut self, val: T) {
        self.data.push(val);
    }

    pub unsafe fn from_matrix(mut mat: Mat<T>, sub_diagonals: c_int, sup_diagonals: c_int) -> BandMat<T> {
        let data = mat.as_mut_ptr();
        let length = mat.cols() * mat.rows();
        BandMat {
            cols: mat.cols(),
            rows: mat.rows(),
            data: Vec::from_raw_parts(data, length, length),
            sub_diagonals: sub_diagonals,
            sup_diagonals: sup_diagonals,
        }
    }
}

impl<T: Clone> BandMat<T> {
    pub fn fill(value: T, n: usize, m: usize) -> BandMat<T> {
        BandMat {
            rows: n,
            cols: m,
            data: repeat(value).take(n * m).collect(),
            sub_diagonals: n as c_int,
            sup_diagonals: m as c_int,
        }
    }
}

impl<T> Index<usize> for BandMat<T> {
    type Output = [T];

    fn index<'a>(&'a self, index: usize) -> &'a [T] {
        let offset = (index * self.cols) as isize;

        unsafe {
            let ptr = (&self.data[..]).as_ptr().offset(offset);
            slice::from_raw_parts(ptr, self.cols)
        }
    }
}

impl<T: fmt::Display> fmt::Display for BandMat<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0usize..self.rows {
            for j in 0usize..self.cols {
                match write!(f, "{}", self[i][j]) {
                    Ok(_) => (),
                    x => return x,
                }
            }

            match writeln!(f, "") {
                Ok(_) => (),
                x => return x,
            }
        }

        Ok(())
    }
}

impl<T> Matrix<T> for BandMat<T> {
    fn rows(&self) -> i32 {
        let n: Option<i32> = NumCast::from(self.rows);
        n.unwrap()
    }

    fn cols(&self) -> i32 {
        let n: Option<i32> = NumCast::from(self.cols);
        n.unwrap()
    }

    fn as_ptr(&self) -> *const T {
        self.data[..].as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        (&mut self.data[..]).as_mut_ptr()
    }
}

impl<T> BandMatrix<T> for BandMat<T> {
    fn sub_diagonals(&self) -> i32 {
        self.sub_diagonals
    }

    fn sup_diagonals(&self) -> i32 {
        self.sup_diagonals
    }

    fn as_matrix(&self) -> &dyn Matrix<T> { self }
}

impl<'a, T> From<&'a dyn BandMatrix<T>> for BandMat<T>
where
    T: Copy,
{
    fn from(a: &dyn BandMatrix<T>) -> BandMat<T> {
        let n = a.rows() as usize;
        let m = a.cols() as usize;
        let len = n * m;

        let sub = a.sub_diagonals() as c_int;
        let sup = a.sup_diagonals() as c_int;

        let mut result = BandMat {
            rows: n,
            cols: m,
            data: Vec::with_capacity(len),
            sub_diagonals: sub,
            sup_diagonals: sup,
        };
        unsafe {
            result.data.set_len(len);
        }

        Copy::copy_mat(a.as_matrix(), &mut result);
        result
    }
}
