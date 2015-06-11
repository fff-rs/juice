// Copyright 2015 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.
#![macro_use]

use std::fmt;
use std::iter::repeat;
use std::mem;
use std::ops::{
    Index,
};
use std::raw;
use num::traits::NumCast;
use Matrix;

#[derive(Debug, PartialEq)]
pub struct Mat<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T> Mat<T> {
    pub fn new() -> Mat<T> {
        Mat {
            rows: 0,
            cols: 0,
            data: Vec::new(),
        }
    }

    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }
    pub unsafe fn set_rows(&mut self, n: usize) { self.rows = n; }
    pub unsafe fn set_cols(&mut self, n: usize) { self.cols = n; }

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

    fn index<'a>(&'a self, index: usize) -> &'a [T] {
        let offset = (index * self.cols) as isize;

        unsafe {
            let ptr = (&self.data[..]).as_ptr().offset(offset);
            mem::transmute(raw::Slice { data: ptr, len: self.cols })
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

            match writeln!(f, "") {
                Ok(_) => (),
                x => return x,
            }
        }

        Ok(())
    }
}

impl<T> Matrix<T> for Mat<T> {
    fn rows(&self) -> i32 {
        let n: Option<i32> = NumCast::from(self.rows);
        n.unwrap()
    }

    fn cols(&self) -> i32 {
        let n: Option<i32> = NumCast::from(self.cols);
        n.unwrap()
    }

    unsafe fn as_ptr(&self) -> *const T {
        self.data[..].as_ptr()
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut T {
        (&mut self.data[..]).as_mut_ptr()
    }
}

#[macro_export]
macro_rules! mat(
    ($($($e: expr),+);*) => ({
        // leading _ to allow empty construction without a warning.
        let mut _temp = Mat::new();
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
    use math::Mat;

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
