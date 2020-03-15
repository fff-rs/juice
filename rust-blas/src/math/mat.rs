// Copyright 2015 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.
#![macro_use]

use crate::vector::ops::Copy;
use crate::Matrix;
use num::traits::NumCast;
use std::fmt;
use std::iter::repeat;
use std::ops::Index;
use std::slice;

#[derive(Debug, PartialEq)]
pub struct Mat<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T> Mat<T> {
    pub fn new_from_data(rows: usize, cols: usize, data: Vec<T>) -> Mat<T> {
        Mat { rows, cols, data }
    }

    pub fn new(n: usize, m: usize) -> Mat<T> {
        let len = n * m;
        let mut data = Vec::with_capacity(len);
        unsafe {
            data.set_len(len);
        }

        Self::new_from_data(n, m, data)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    /// Set Matrix Rows Manually
    /// # Safety
    /// This only sets the value for rows, and does not include any guarantees that the
    /// number of elements is equal to rows x columns after the operation takes place.
    pub unsafe fn set_rows(&mut self, n: usize) {
        self.rows = n;
    }
    /// Set Matrix Columns Manually
    /// # Safety
    /// This only sets the value for columns, and does not include any guarantees that the
    /// number of elements is equal to rows x columns after the operation takes place.
    pub unsafe fn set_cols(&mut self, n: usize) {
        self.cols = n;
    }
    /// Push a single value to matrix data backing
    /// # Safety
    /// This makes no checks that rows x columns remains equivalent to the length of pushed elements
    pub unsafe fn push(&mut self, val: T) {
        self.data.push(val);
    }
}

impl<T: Clone> Mat<T> {
    pub fn fill(value: T, n: usize, m: usize) -> Mat<T> {
        Mat {
            rows: n,
            cols: m,
            data: repeat(value).take(n * m).collect(),
        }
    }
}

impl<T> Index<usize> for Mat<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &[T] {
        let offset = (index * self.cols) as isize;

        unsafe {
            let ptr = (&self.data[..]).as_ptr().offset(offset);
            slice::from_raw_parts(ptr, self.cols)
        }
    }
}

impl<T: fmt::Display> fmt::Display for Mat<T> {
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

impl<T> Matrix<T> for Mat<T> {
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

impl<'a, T> From<&'a dyn Matrix<T>> for Mat<T>
where
    T: Copy,
{
    fn from(a: &dyn Matrix<T>) -> Mat<T> {
        let n = a.rows() as usize;
        let m = a.cols() as usize;
        let len = n * m;

        let mut result = Mat {
            rows: n,
            cols: m,
            data: Vec::with_capacity(len),
        };
        unsafe {
            result.data.set_len(len);
        }

        Copy::copy_mat(a, &mut result);
        result
    }
}

#[macro_export]
macro_rules! mat(
    ($($($e: expr),+);*) => ({
        // leading _ to allow empty construction without a warning.
        let mut _temp = Mat::new(0, 0);
        let mut rows = 0usize;
        let mut _cols;
        $(
            rows += 1;
            _cols = 0usize;
            $(
                _cols += 1;
                unsafe {
                    _temp.push($e);
                }
            )+
        )*

        unsafe {
            _temp.set_rows(rows);
            _temp.set_cols(_cols);
        }

        _temp
    });
);

#[cfg(test)]
mod tests {
    use crate::math::Mat;

    #[test]
    fn index() {
        let a = mat![1f32, 2f32];
        assert_eq!(1.0, a[0][0]);
        assert_eq!(2.0, a[0][1]);

        let b = mat![1f32; 2f32];
        assert_eq!(1.0, b[0][0]);
        assert_eq!(2.0, b[1][0]);

        let m = mat![1f32, 2f32; 3f32, 4f32];
        assert_eq!(1.0, m[0][0]);
        assert_eq!(2.0, m[0][1]);
        assert_eq!(3.0, m[1][0]);
        assert_eq!(4.0, m[1][1]);
    }
}
