// Copyright 2015 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.
use crate::math::Mat;
use crate::matrix::BandMatrix;
use crate::vector::ops::Copy;
use crate::Matrix;
use num::traits::NumCast;
use std::fmt;
use std::iter::repeat;
use std::ops::Index;
use std::slice;

#[derive(Debug, PartialEq)]
/// Banded Matrix
/// A banded matrix is a matrix where only the diagonal, a number of super-diagonals and a number of
/// sub-diagonals are non-zero.
/// https://en.wikipedia.org/wiki/Band_matrix
pub struct BandMat<T> {
    rows: usize,
    cols: usize,
    sub_diagonals: u32,
    sup_diagonals: u32,
    data: Vec<T>,
}

impl<T> BandMat<T> {
    pub fn new(n: usize, m: usize, sub: u32, sup: u32) -> BandMat<T> {
        let len = n * m;
        let mut data = Vec::with_capacity(len);
        unsafe {
            data.set_len(len);
        }

        BandMat {
            rows: n,
            cols: m,
            data,
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
    /// Set Rows Manually
    /// # Safety
    /// No guarantees are made about rows x columns being equivalent to data length after this
    /// operation
    pub unsafe fn set_rows(&mut self, n: usize) {
        self.rows = n;
    }
    /// Set Columns Manually
    /// # Safety
    /// No guarantees are made about rows x columns being equivalent to data length after this
    /// operation
    pub unsafe fn set_cols(&mut self, n: usize) {
        self.cols = n;
    }
    pub unsafe fn set_sub_diagonals(&mut self, n: u32) {
        self.sub_diagonals = n;
    }
    pub unsafe fn set_sup_diagonals(&mut self, n: u32) {
        self.sup_diagonals = n;
    }

    pub unsafe fn push(&mut self, val: T) {
        self.data.push(val);
    }

    pub unsafe fn from_matrix(
        mut mat: Mat<T>,
        sub_diagonals: u32,
        sup_diagonals: u32,
    ) -> BandMat<T> {
        let data = mat.as_mut_ptr();
        let length = mat.cols() * mat.rows();
        BandMat {
            cols: mat.cols(),
            rows: mat.rows(),
            data: Vec::from_raw_parts(data, length, length),
            sub_diagonals,
            sup_diagonals,
        }
    }
}

impl<T: Clone> BandMat<T> {
    pub fn fill(value: T, n: usize, m: usize) -> BandMat<T> {
        BandMat {
            rows: n,
            cols: m,
            data: repeat(value).take(n * m).collect(),
            sub_diagonals: n as u32,
            sup_diagonals: m as u32,
        }
    }
}

impl<T> Index<usize> for BandMat<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &[T] {
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

            match writeln!(f) {
                Ok(_) => (),
                x => return x,
            }
        }

        Ok(())
    }
}

impl<T> Matrix<T> for BandMat<T> {
    fn rows(&self) -> u32 {
        let n: Option<u32> = NumCast::from(self.rows);
        n.unwrap()
    }

    fn cols(&self) -> u32 {
        let n: Option<u32> = NumCast::from(self.cols);
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
    fn sub_diagonals(&self) -> u32 {
        self.sub_diagonals
    }

    fn sup_diagonals(&self) -> u32 {
        self.sup_diagonals
    }

    fn as_matrix(&self) -> &dyn Matrix<T> {
        self
    }
}

impl<'a, T> From<&'a dyn BandMatrix<T>> for BandMat<T>
where
    T: Copy,
{
    fn from(a: &dyn BandMatrix<T>) -> BandMat<T> {
        let n = a.rows() as usize;
        let m = a.cols() as usize;
        let len = n * m;

        let sub = a.sub_diagonals() as u32;
        let sup = a.sup_diagonals() as u32;

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
