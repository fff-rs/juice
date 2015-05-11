// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

//! Bindings for matrix-vector functions.

use libc::{c_double, c_float, c_int, c_void};

use attribute::{
    Order,
    Transpose,
    Symmetry,
    Diagonal,
};

#[link(name = "blas")]
extern {
    // Multiply
    pub fn cblas_sgemv(order: Order, trans: Transpose, m: c_int, n: c_int, alpha: c_float,       a: *const c_float,  lda: c_int, x: *const c_float,  inc_x: c_int, beta: c_float,       y: *mut c_float,  inc_y: c_int);
    pub fn cblas_dgemv(order: Order, trans: Transpose, m: c_int, n: c_int, alpha: c_double,      a: *const c_double, lda: c_int, x: *const c_double, inc_x: c_int, beta: c_double,      y: *mut c_double, inc_y: c_int);
    pub fn cblas_cgemv(order: Order, trans: Transpose, m: c_int, n: c_int, alpha: *const c_void, a: *const c_void,   lda: c_int, x: *const c_void,   inc_x: c_int, beta: *const c_void, y: *mut c_void,   inc_y: c_int);
    pub fn cblas_zgemv(order: Order, trans: Transpose, m: c_int, n: c_int, alpha: *const c_void, a: *const c_void,   lda: c_int, x: *const c_void,   inc_x: c_int, beta: *const c_void, y: *mut c_void,   inc_y: c_int);

    pub fn cblas_ssymv(order: Order, sym: Symmetry, n: c_int, alpha: c_float,       a: *const c_float,  lda: c_int, x: *const c_float,  inc_x: c_int, beta: c_float,       y: *mut c_float,  inc_y: c_int);
    pub fn cblas_dsymv(order: Order, sym: Symmetry, n: c_int, alpha: c_double,      a: *const c_double, lda: c_int, x: *const c_double, inc_x: c_int, beta: c_double,      y: *mut c_double, inc_y: c_int);
    pub fn cblas_csymv(order: Order, sym: Symmetry, n: c_int, alpha: *const c_void, a: *const c_void,   lda: c_int, x: *const c_void,   inc_x: c_int, beta: *const c_void, y: *mut c_void,   inc_y: c_int);
    pub fn cblas_zsymv(order: Order, sym: Symmetry, n: c_int, alpha: *const c_void, a: *const c_void,   lda: c_int, x: *const c_void,   inc_x: c_int, beta: *const c_void, y: *mut c_void,   inc_y: c_int);

    pub fn cblas_chemv(order: Order, sym: Symmetry, n: c_int, alpha: *const c_void, a: *const c_void,   lda: c_int, x: *const c_void,   inc_x: c_int, beta: *const c_void, y: *mut c_void,   inc_y: c_int);
    pub fn cblas_zhemv(order: Order, sym: Symmetry, n: c_int, alpha: *const c_void, a: *const c_void,   lda: c_int, x: *const c_void,   inc_x: c_int, beta: *const c_void, y: *mut c_void,   inc_y: c_int);

    pub fn cblas_strmv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, a: *const c_float,  lda: c_int, x: *mut c_float,  inc_x: c_int);
    pub fn cblas_dtrmv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, a: *const c_double, lda: c_int, x: *mut c_double, inc_x: c_int);
    pub fn cblas_ctrmv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, a: *const c_void,   lda: c_int, x: *mut c_void,   inc_x: c_int);
    pub fn cblas_ztrmv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, a: *const c_void,   lda: c_int, x: *mut c_void,   inc_x: c_int);

    pub fn cblas_strsv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, a: *const c_float,  lda: c_int, x: *mut c_float,  inc_x: c_int);
    pub fn cblas_dtrsv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, a: *const c_double, lda: c_int, x: *mut c_double, inc_x: c_int);
    pub fn cblas_ctrsv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, a: *const c_void,   lda: c_int, x: *mut c_void,   inc_x: c_int);
    pub fn cblas_ztrsv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, a: *const c_void,   lda: c_int, x: *mut c_void,   inc_x: c_int);

    // Rank Update
    pub fn cblas_sger (order: Order, m: c_int, n: c_int, alpha: c_float,       x: *const c_float,  inc_x: c_int, y: *const c_float,  inc_y: c_int, a: *mut c_float,  lda: c_int);
    pub fn cblas_dger (order: Order, m: c_int, n: c_int, alpha: c_double,      x: *const c_double, inc_x: c_int, y: *const c_double, inc_y: c_int, a: *mut c_double, lda: c_int);

    pub fn cblas_cgeru(order: Order, m: c_int, n: c_int, alpha: *const c_void, x: *const c_void,   inc_x: c_int, y: *const c_void,   inc_y: c_int, a: *mut c_void,   lda: c_int);
    pub fn cblas_zgeru(order: Order, m: c_int, n: c_int, alpha: *const c_void, x: *const c_void,   inc_x: c_int, y: *const c_void,   inc_y: c_int, a: *mut c_void,   lda: c_int);

    pub fn cblas_cgerc(order: Order, m: c_int, n: c_int, alpha: *const c_void, x: *const c_void,   inc_x: c_int, y: *const c_void,   inc_y: c_int, a: *mut c_void,   lda: c_int);
    pub fn cblas_zgerc(order: Order, m: c_int, n: c_int, alpha: *const c_void, x: *const c_void,   inc_x: c_int, y: *const c_void,   inc_y: c_int, a: *mut c_void,   lda: c_int);

    pub fn cblas_cher(order: Order, sym: Symmetry, n: c_int, alpha: c_float,  x: *const c_void,   inc_x: c_int, a: *mut c_void,   lda: c_int);
    pub fn cblas_zher(order: Order, sym: Symmetry, n: c_int, alpha: c_double, x: *const c_void,   inc_x: c_int, a: *mut c_void,   lda: c_int);

    pub fn cblas_ssyr(order: Order, sym: Symmetry, n: c_int, alpha: c_float,  x: *const c_float,  inc_x: c_int, a: *mut c_float,  lda: c_int);
    pub fn cblas_dsyr(order: Order, sym: Symmetry, n: c_int, alpha: c_double, x: *const c_double, inc_x: c_int, a: *mut c_double, lda: c_int);

    pub fn cblas_cher2(order: Order, sym: Symmetry, n: c_int, alpha: *const c_void, x: *const c_void,   inc_x: c_int, y: *const c_void,   inc_y: c_int, a: *mut c_void,   lda: c_int);
    pub fn cblas_zher2(order: Order, sym: Symmetry, n: c_int, alpha: *const c_void, x: *const c_void,   inc_x: c_int, y: *const c_void,   inc_y: c_int, a: *mut c_void,   lda: c_int);

    pub fn cblas_ssyr2(order: Order, sym: Symmetry, n: c_int, alpha: c_float,       x: *const c_float,  inc_x: c_int, y: *const c_float,  inc_y: c_int, a: *mut c_float,  lda: c_int);
    pub fn cblas_dsyr2(order: Order, sym: Symmetry, n: c_int, alpha: c_double,      x: *const c_double, inc_x: c_int, y: *const c_double, inc_y: c_int, a: *mut c_double, lda: c_int);

    // Band Multiply
    pub fn cblas_sgbmv(order: Order, trans: Transpose, m: c_int, n: c_int, kl: c_int, ku: c_int, alpha: c_float,       a: *const c_float,  lda: c_int, x: *const c_float,  inc_x: c_int, beta: c_float,       y: *mut c_float,  inc_y: c_int);
    pub fn cblas_dgbmv(order: Order, trans: Transpose, m: c_int, n: c_int, kl: c_int, ku: c_int, alpha: c_double,      a: *const c_double, lda: c_int, x: *const c_double, inc_x: c_int, beta: c_double,      y: *mut c_double, inc_y: c_int);
    pub fn cblas_cgbmv(order: Order, trans: Transpose, m: c_int, n: c_int, kl: c_int, ku: c_int, alpha: *const c_void, a: *const c_void,   lda: c_int, x: *const c_void,   inc_x: c_int, beta: *const c_void, y: *mut c_void,   inc_y: c_int);
    pub fn cblas_zgbmv(order: Order, trans: Transpose, m: c_int, n: c_int, kl: c_int, ku: c_int, alpha: *const c_void, a: *const c_void,   lda: c_int, x: *const c_void,   inc_x: c_int, beta: *const c_void, y: *mut c_void,   inc_y: c_int);

    pub fn cblas_chbmv(order: Order, sym: Symmetry, n: c_int, k: c_int, alpha: *const c_void, a: *const c_void,   lda: c_int, x: *const c_void,   inc_x: c_int, beta: *const c_void, y: *mut c_void,   inc_y: c_int);
    pub fn cblas_zhbmv(order: Order, sym: Symmetry, n: c_int, k: c_int, alpha: *const c_void, a: *const c_void,   lda: c_int, x: *const c_void,   inc_x: c_int, beta: *const c_void, y: *mut c_void,   inc_y: c_int);

    pub fn cblas_ssbmv(order: Order, sym: Symmetry, n: c_int, k: c_int, alpha: c_float,       a: *const c_float,  lda: c_int, x: *const c_float,  inc_x: c_int, beta: c_float,       y: *mut c_float,  inc_y: c_int);
    pub fn cblas_dsbmv(order: Order, sym: Symmetry, n: c_int, k: c_int, alpha: c_double,      a: *const c_double, lda: c_int, x: *const c_double, inc_x: c_int, beta: c_double,      y: *mut c_double, inc_y: c_int);

    pub fn cblas_stbmv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, k: c_int, a: *const c_float,  x: *mut c_float,  inc_x: c_int);
    pub fn cblas_dtbmv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, k: c_int, a: *const c_double, x: *mut c_double, inc_x: c_int);
    pub fn cblas_ctbmv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, k: c_int, a: *const c_void,   x: *mut c_void,   inc_x: c_int);
    pub fn cblas_ztbmv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, k: c_int, a: *const c_void,   x: *mut c_void,   inc_x: c_int);

    pub fn cblas_stbsv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, k: c_int, a: *const c_float,  x: *mut c_float,  inc_x: c_int);
    pub fn cblas_dtbsv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, k: c_int, a: *const c_double, x: *mut c_double, inc_x: c_int);
    pub fn cblas_ctbsv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, k: c_int, a: *const c_void,   x: *mut c_void,   inc_x: c_int);
    pub fn cblas_ztbsv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, k: c_int, a: *const c_void,   x: *mut c_void,   inc_x: c_int);

    // Packed Multiply
    pub fn cblas_chpmv(order: Order, sym: Symmetry, n: c_int, alpha: *const c_void, a: *const c_void,   x: *const c_void,   inc_x: c_int, beta: *const c_void, y: *mut c_void,   inc_y: c_int);
    pub fn cblas_zhpmv(order: Order, sym: Symmetry, n: c_int, alpha: *const c_void, a: *const c_void,   x: *const c_void,   inc_x: c_int, beta: *const c_void, y: *mut c_void,   inc_y: c_int);

    pub fn cblas_sspmv(order: Order, sym: Symmetry, n: c_int, alpha: c_float,       a: *const c_float,  x: *const c_float,  inc_x: c_int, beta: c_float,       y: *mut c_float,  inc_y: c_int);
    pub fn cblas_dspmv(order: Order, sym: Symmetry, n: c_int, alpha: c_double,      a: *const c_double, x: *const c_double, inc_x: c_int, beta: c_double,      y: *mut c_double, inc_y: c_int);

    pub fn cblas_stpmv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, a: *const c_float,  x: *mut c_float,  inc_x: c_int);
    pub fn cblas_dtpmv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, a: *const c_double, x: *mut c_double, inc_x: c_int);
    pub fn cblas_ctpmv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, a: *const c_void,   x: *mut c_void,   inc_x: c_int);
    pub fn cblas_ztpmv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, a: *const c_void,   x: *mut c_void,   inc_x: c_int);

    pub fn cblas_stpsv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, a: *const c_float,  x: *mut c_float,  inc_x: c_int);
    pub fn cblas_dtpsv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, a: *const c_double, x: *mut c_double, inc_x: c_int);
    pub fn cblas_ctpsv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, a: *const c_void,   x: *mut c_void,   inc_x: c_int);
    pub fn cblas_ztpsv(order: Order, sym: Symmetry, trans: Transpose, diag: Diagonal, n: c_int, a: *const c_void,   x: *mut c_void,   inc_x: c_int);

    // Packed Rank Update
    pub fn cblas_chpr(order: Order, sym: Symmetry, n: c_int, alpha: c_float,  x: *const c_void,   inc_x: c_int, a: *mut c_void);
    pub fn cblas_zhpr(order: Order, sym: Symmetry, n: c_int, alpha: c_double, x: *const c_void,   inc_x: c_int, a: *mut c_void);

    pub fn cblas_sspr(order: Order, sym: Symmetry, n: c_int, alpha: c_float,  x: *const c_float,  inc_x: c_int, a: *mut c_float);
    pub fn cblas_dspr(order: Order, sym: Symmetry, n: c_int, alpha: c_double, x: *const c_double, inc_x: c_int, a: *mut c_double);

    pub fn cblas_chpr2(order: Order, sym: Symmetry, n: c_int, alpha: *const c_void, x: *const c_void,   inc_x: c_int, y: *const c_void,   inc_y: c_int, a: *mut c_void);
    pub fn cblas_zhpr2(order: Order, sym: Symmetry, n: c_int, alpha: *const c_void, x: *const c_void,   inc_x: c_int, y: *const c_void,   inc_y: c_int, a: *mut c_void);

    pub fn cblas_sspr2(order: Order, sym: Symmetry, n: c_int, alpha: c_float,       x: *const c_float,  inc_x: c_int, y: *const c_float,  inc_y: c_int, a: *mut c_float);
    pub fn cblas_dspr2(order: Order, sym: Symmetry, n: c_int, alpha: c_double,      x: *const c_double, inc_x: c_int, y: *const c_double, inc_y: c_int, a: *mut c_double);
}
