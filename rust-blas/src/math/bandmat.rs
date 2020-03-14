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
use std::mem::ManuallyDrop;
use std::cmp::{max, min};

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
    original_dims: Vec<usize>,
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
            original_dims: vec![n, m],
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

}

impl<T: std::marker::Copy> BandMat<T> {
    /// Converts a standard matrix into a band matrix
    ///
    /// TODO: write a conversion utility
    /// DOES NOT PERFORM CONVERSION INTO BAND MATRIX FORMAT
    ///
    /// The actual conversion can be done by hand once one
    /// understands the pattern of the band matrix. Say you
    /// have an MxN band matrix in standard form. Say you 
    /// have ku upper diagonals and kl sub diagonals. Then
    /// the band matrix representation will have dimensions 
    /// Nx(ku+kl+1), and will have the all of the relevant 
    /// values of each row, concatenated starting at position
    /// kl. An example is probably useful:
    ///
    /// Say our initial matrix is the following:
    ///
    /// [ 1 2 0 0 ]
    /// [ 3 1 2 0 ]
    /// [ 0 3 1 2 ]
    /// [ 0 0 3 1 ]
    ///
    /// Then the band matrix representation would be:
    ///
    /// [ x 1 2 ]
    /// [ 3 1 2 ]
    /// [ 3 1 2 ]
    /// [ 3 1 x ]
    ///
    /// "x" represent values that aren't relevant to the 
    /// calculations: the values in those slots won't be
    /// read or modified by the underlying BLAS program.
    ///
    pub fn from_matrix(
        mat: Mat<T>,
        sub_diagonals: u32,
        sup_diagonals: u32,
    ) -> BandMat<T> 
    where T: std::fmt::Debug
    {
        let original_dims = vec![mat.rows(), mat.cols()];
        let mut mat = ManuallyDrop::new(mat);
        
        let cols = mat.cols();
        let rows = mat.rows();
        let lda = (sub_diagonals + 1 + sup_diagonals) as usize;
        let mut v = unsafe {
            let length = rows * cols;
            Vec::from_raw_parts(mat.as_mut_ptr(), length, length)
        };


        /*
        let mut i = (sup_diagonals) as usize;
        let square_dim = rows;
        for r in 0..square_dim {
            let start = (r * original_cols) + max(0, r as isize - (sub_diagonals as isize)) as usize;
            let end = (r * original_cols) + min(square_dim, r + (sup_diagonals as usize) + 1usize);
            (&mut v).copy_within(start..end, i);
            i = i + (end - start);
        }
        */

        //println!("{:?}", v);

        for r in 0..rows {
            let s = (r * cols) + max(0, r as isize - sub_diagonals as isize) as usize;
            let e = (r * cols) + min(cols, r + sup_diagonals as usize + 1usize);

            let offset = max(0, (lda as isize) - sup_diagonals as isize - r as isize - 1) as usize;
            let i = (r * lda) + offset;
            let i = i as usize;
            (&mut v).copy_within(s..e, i); 
        }

        BandMat {
            cols,
            rows,
            data: v,
            sub_diagonals,
            sup_diagonals,
            original_dims,
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
            original_dims: vec![n as usize, m as usize],
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
    fn lead_dim(&self) -> u32 {
        self.sub_diagonals + self.sup_diagonals + 1
    }

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

    fn original_dims(&self) -> &Vec<usize> {
        self.original_dims.as_ref()
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
            original_dims: a.original_dims().clone(),
        };
        unsafe {
            result.data.set_len(len);
        }

        Copy::copy_mat(a.as_matrix(), &mut result);
        result
    }
}

mod tests {
    use super::*;

    fn write_to_memory<T: Clone>(dest: *mut T, source: &Vec<T>) -> () {
        let mut v1 = vec![];
        unsafe {
            v1 = Vec::from_raw_parts(dest, source.len(), source.len());
            v1.clone_from(source);
        }
        let _ = ManuallyDrop::new(v1);
    }

    fn retrieve_memory<T: Clone>(t: &mut dyn Matrix<T>, l: usize) -> Vec<T> {
        let mut v: Vec<T> = vec![];

        unsafe {
            let v1 = Vec::from_raw_parts(t.as_mut_ptr(), l, l);
            v.clone_from(&v1);
            let _ = ManuallyDrop::new(v1);
        }

        v
    }

    #[test]
    fn basic_conversion_test() {
        let v: Vec<f32> = vec![
           0.5, 2.0, 0.0, 0.0,
           2.0, 0.5, 2.0, 0.0,
           0.0, 2.0, 0.5, 2.0,
           0.0, 0.0, 2.0, 0.5,
        ];

        let mut m: Mat<f32> = Mat::new(4, 4);
        let length = m.rows() * m.cols();

        write_to_memory(m.as_mut_ptr(), &v);

        let mut band_m = BandMat::from_matrix(m, 1, 1); 

        println!("{:?}", retrieve_memory(&mut band_m, length));

    }
}
