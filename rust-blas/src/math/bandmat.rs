// Copyright 2015 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.
use std::fmt::Debug;
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

}

impl<T: std::marker::Copy> BandMat<T> {
    /// Converts a standard matrix into a band matrix
    pub fn from_matrix(
        mat: Mat<T>,
        sub_diagonals: u32,
        sup_diagonals: u32,
    ) -> BandMat<T> 
    {
        let mut mat = ManuallyDrop::new(mat);
        
        let cols = mat.cols();
        let rows = mat.rows();
        let lda = (sub_diagonals + 1 + sup_diagonals) as usize;
        let mut v = unsafe {
            let length = rows * cols;
            Vec::from_raw_parts(mat.as_mut_ptr(), length, length)
        };

        /*
         * TODO: Write comment explaining what's going on here
         *
         */
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
        }
    }

}

impl<T: std::marker::Copy + Default> BandMat<T> {
    pub fn to_matrix(bandmat: Self) -> Mat<T> {
        let mut bandmat = ManuallyDrop::new(bandmat);

        let ku = bandmat.sup_diagonals() as usize;
        let kl = bandmat.sub_diagonals() as usize;
        let lda = ku + kl + 1;
        let rows = bandmat.rows();
        let cols = bandmat.cols();

        let mut v = unsafe {
            let length = rows * cols;
            Vec::from_raw_parts(bandmat.as_mut_ptr(), length, length)
        };


        let num_of_last_row_terms = kl + 1 - (rows - min(rows, cols));

        for r in (0..rows).rev() {
            let offset = rows - r - 1;

            let s = max(0, -(kl as isize + 1) + (num_of_last_row_terms -(if rows > cols {1} else {2}) ) as isize + offset as isize);
            let s = (r * lda) as isize + s;
            let s = s as usize; 

            let e = min(lda, num_of_last_row_terms + offset);
            let e = (r * lda) + e;

            let p = cols as isize - num_of_last_row_terms as isize - offset as isize;
            let i = (r * cols) + max(0, p) as usize;

            v.copy_within(s..e, i);

            let l = e - s;
            let zero_range = (r*cols)..max(0, i);
            let zero_range = zero_range.chain(min((r+1)*cols, i+l)..((r+1)*cols));
            for i in zero_range {
                v[i] = T::default();
            }
        }

        Mat::new_from_data(rows, cols, v)
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
           1.0, 0.5, 2.0, 0.0,
           0.0, 1.0, 0.5, 2.0,
           0.0, 0.0, 1.0, 0.5,
        ];

        let mut m: Mat<f32> = Mat::new(4, 4);
        let length = m.rows() * m.cols();

        write_to_memory(m.as_mut_ptr(), &v);

        let mut band_m = BandMat::from_matrix(m, 1, 1); 

        let result_vec = retrieve_memory(&mut band_m, length);

        // Check random values in position to make sure that they're correct, since it's hard to
        // actualy predict the real vector values
        assert_eq!(result_vec[1], 0.5f32);
        assert_eq!(result_vec[2], 2.0f32);
        assert_eq!(result_vec[3], 1.0f32);
        assert_eq!(result_vec[7], 0.5f32);
        assert_eq!(result_vec[9], 1.0f32);
    }

    #[test]
    fn nonsquare_conversion_test() {
        let v: Vec<f32> = vec![
            0.5, 1.0, 0.0, 0.0,
            2.0, 0.5, 1.0, 0.0,
            3.0, 2.0, 0.5, 1.0,
            0.0, 3.0, 2.0, 0.5,
            0.0, 0.0, 3.0, 2.0,
            0.0, 0.0, 0.0, 3.0,
        ];

        let mut m: Mat<f32> = Mat::new(6, 4);
        let length = m.rows() * m.cols();

        write_to_memory(m.as_mut_ptr(), &v);

        let mut band_m = BandMat::from_matrix(m, 2, 1);

        let result_vec = retrieve_memory(&mut band_m, length);

        assert_eq!(result_vec[2], 0.5);
        assert_eq!(result_vec[5], 2.0);
        assert_eq!(result_vec[7], 1.0);
        assert_eq!(result_vec[8], 3.0);
        assert_eq!(result_vec[16], 3.0);
        assert_eq!(result_vec[20], 3.0);
    }

    #[test]
    fn to_and_from_conversion_test() {
        let original: Vec<f32> = vec![
            0.5, 2.0, 0.0, 0.0,
            1.0, 0.5, 2.0, 0.0,
            0.0, 1.0, 0.5, 2.0,
            0.0, 0.0, 1.0, 0.5,
        ];
        let v = original.clone(); 

        let mut m: Mat<f32> = Mat::new(4, 4);
        let length = m.rows() * m.cols();

        write_to_memory(m.as_mut_ptr(), &v);

        let band_m = BandMat::from_matrix(m, 1, 1);
        let mut m = BandMat::to_matrix(band_m);

        let result_vec = retrieve_memory(&mut m, length);

        assert_eq!(result_vec, original);
    }

    #[test]
    fn to_and_from_nonsquare_test() {
        let original: Vec<f32> = vec![
            0.5, 1.0, 0.0, 0.0,
            2.0, 0.5, 1.0, 0.0,
            3.0, 2.0, 0.5, 1.0,
            0.0, 3.0, 2.0, 0.5,
            0.0, 0.0, 3.0, 2.0,
        ];
        let v = original.clone();

        let mut m: Mat<f32> = Mat::new(5, 4);
        let length = m.rows() * m.cols();

        write_to_memory(m.as_mut_ptr(), &v);

        let band_m = BandMat::from_matrix(m, 2, 1);
        let mut m = BandMat::to_matrix(band_m);

        let result_vec = retrieve_memory(&mut m, length);

        assert_eq!(result_vec, original);
    }

    #[test]
    fn to_and_from_nonsquare2_test() {
        let original: Vec<f32> = vec![
            0.5, 1.0, 0.0, 0.0,
            2.0, 0.5, 1.0, 0.0,
            3.0, 2.0, 0.5, 1.0,
            0.0, 3.0, 2.0, 0.5,
            0.0, 0.0, 3.0, 2.0,
            0.0, 0.0, 0.0, 3.0,
        ];
        let v = original.clone();

        let mut m: Mat<f32> = Mat::new(6, 4);
        let length = m.rows() * m.cols();

        write_to_memory(m.as_mut_ptr(), &v);

        let band_m = BandMat::from_matrix(m, 2, 1);
        let mut m = BandMat::to_matrix(band_m);

        let result_vec = retrieve_memory(&mut m, length);

        assert_eq!(result_vec, original);
    }
}
