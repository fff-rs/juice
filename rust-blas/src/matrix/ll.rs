// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

//! Bindings for matrix functions.

pub mod cblas_s {
    use libc::{c_float};
    use crate::attribute::{
        Order,
        Transpose,
        Symmetry,
        Diagonal,
        Side,
    };

    pub use self::cblas_sgemm as gemm;
    pub use self::cblas_ssymm as symm;
    pub use self::cblas_ssyr2k as syr2k;
    pub use self::cblas_ssyrk as syrk;
    pub use self::cblas_strmm as trmm;
    pub use self::cblas_strsm as strsm;

    extern {
        pub fn cblas_sgemm(order: Order, trans_a: Transpose, trans_b: Transpose, m: u32, n: u32, k: u32, alpha: c_float,       a: *const c_float,  lda: u32, b: *const c_float,  ldb: u32, beta: c_float,       c: *mut c_float,  ldc: u32);
        pub fn cblas_ssymm(order: Order, side: Side, sym: Symmetry, m: u32, n: u32, alpha: c_float,       a: *const c_float,  lda: u32, b: *const c_float,  ldb: u32, beta: c_float,       c: *mut c_float,  ldc: u32);
        pub fn cblas_strmm(order: Order, side: Side, sym: Symmetry, trans: Transpose, diag: Diagonal, m: u32, n: u32, alpha: c_float,       a: *const c_float,  lda: u32, b: *mut c_float,  ldb: u32);
        pub fn cblas_strsm(order: Order, side: Side, sym: Symmetry, trans: Transpose, diag: Diagonal, m: u32, n: u32, alpha: c_float,       a: *const c_float,  lda: u32, b: *mut c_float,  ldb: u32);
        pub fn cblas_ssyrk(order: Order, sym: Symmetry, Trans: Transpose, n: u32, k: u32, alpha: c_float,       a: *const c_float,  lda: u32, beta: c_float,       c: *mut c_float,  ldc: u32);
        pub fn cblas_ssyr2k(order: Order, sym: Symmetry, Trans: Transpose, n: u32, k: u32, alpha: c_float,       a: *const c_float,  lda: u32, b: *const c_float,  ldb: u32, beta: c_float,       c: *mut c_float,  ldc: u32);
    }
}

pub mod cblas_d {
    use libc::{c_double};
    use crate::attribute::{
        Order,
        Transpose,
        Symmetry,
        Diagonal,
        Side,
    };

    pub use self::cblas_dgemm as gemm;
    pub use self::cblas_dsymm as symm;
    pub use self::cblas_dsyr2k as syr2k;
    pub use self::cblas_dsyrk as syrk;
    pub use self::cblas_dtrmm as trmm;
    pub use self::cblas_dtrsm as strsm;

    extern {
        pub fn cblas_dgemm(order: Order, trans_a: Transpose, trans_b: Transpose, m: u32, n: u32, k: u32, alpha: c_double,       a: *const c_double,  lda: u32, b: *const c_double,  ldb: u32, beta: c_double,       c: *mut c_double,  ldc: u32);
        pub fn cblas_dsymm(order: Order, side: Side, sym: Symmetry, m: u32, n: u32, alpha: c_double,       a: *const c_double,  lda: u32, b: *const c_double,  ldb: u32, beta: c_double,       c: *mut c_double,  ldc: u32);
        pub fn cblas_dtrmm(order: Order, side: Side, sym: Symmetry, trans: Transpose, diag: Diagonal, m: u32, n: u32, alpha: c_double,       a: *const c_double,  lda: u32, b: *mut c_double,  ldb: u32);
        pub fn cblas_dtrsm(order: Order, side: Side, sym: Symmetry, trans: Transpose, diag: Diagonal, m: u32, n: u32, alpha: c_double,       a: *const c_double,  lda: u32, b: *mut c_double,  ldb: u32);
        pub fn cblas_dsyrk(order: Order, sym: Symmetry, Trans: Transpose, n: u32, k: u32, alpha: c_double,       a: *const c_double,  lda: u32, beta: c_double,       c: *mut c_double,  ldc: u32);
        pub fn cblas_dsyr2k(order: Order, sym: Symmetry, Trans: Transpose, n: u32, k: u32, alpha: c_double,       a: *const c_double,  lda: u32, b: *const c_double,  ldb: u32, beta: c_double,       c: *mut c_double,  ldc: u32);
    }
}

pub mod cblas_c {
    use libc::{c_float, c_void};
    use crate::attribute::{
        Order,
        Transpose,
        Symmetry,
        Diagonal,
        Side,
    };

    pub use self::cblas_cgemm as gemm;
    pub use self::cblas_chemm as hemm;
    pub use self::cblas_cher2k as her2k;
    pub use self::cblas_cherk as herk;
    pub use self::cblas_csymm as symm;
    pub use self::cblas_csyr2k as syr2k;
    pub use self::cblas_csyrk as syrk;
    pub use self::cblas_ctrmm as trmm;
    pub use self::cblas_ctrsm as trsm;

    extern {
        pub fn cblas_cgemm(order: Order, trans_a: Transpose, trans_b: Transpose, m: u32, n: u32, k: u32, alpha: *const c_void, a: *const c_void,   lda: u32, b: *const c_void,   ldb: u32, beta: *const c_void, c: *mut c_void,   ldc: u32);
        pub fn cblas_csymm(order: Order, side: Side, sym: Symmetry, m: u32, n: u32, alpha: *const c_void, a: *const c_void,   lda: u32, b: *const c_void,   ldb: u32, beta: *const c_void, c: *mut c_void,   ldc: u32);
        pub fn cblas_chemm(order: Order, side: Side, sym: Symmetry, m: u32, n: u32, alpha: *const c_void, a: *const c_void,   lda: u32, b: *const c_void,   ldb: u32, beta: *const c_void, c: *mut c_void,   ldc: u32);
        pub fn cblas_ctrmm(order: Order, side: Side, sym: Symmetry, trans: Transpose, diag: Diagonal, m: u32, n: u32, alpha: *const c_void, a: *const c_void,   lda: u32, b: *mut c_void,   ldb: u32);
        pub fn cblas_ctrsm(order: Order, side: Side, sym: Symmetry, trans: Transpose, diag: Diagonal, m: u32, n: u32, alpha: *const c_void, a: *const c_void,   lda: u32, b: *mut c_void,   ldb: u32);
        pub fn cblas_cherk(order: Order, sym: Symmetry, Trans: Transpose, n: u32, k: u32, alpha: c_float,  a: *const c_void, lda: u32, beta: c_float,  c: *mut c_void, ldc: u32);
        pub fn cblas_cher2k(order: Order, sym: Symmetry, Trans: Transpose, n: u32, k: u32, alpha: *const c_void, a: *const c_void, lda: u32, b: *const c_void, ldb: u32, beta: c_float,  c: *mut c_void, ldc: u32);
        pub fn cblas_csyrk(order: Order, sym: Symmetry, Trans: Transpose, n: u32, k: u32, alpha: *const c_void, a: *const c_void,   lda: u32, beta: *const c_void, c: *mut c_void,   ldc: u32);
        pub fn cblas_csyr2k(order: Order, sym: Symmetry, Trans: Transpose, n: u32, k: u32, alpha: *const c_void, a: *const c_void,   lda: u32, b: *const c_void,   ldb: u32, beta: *const c_void, c: *mut c_void,   ldc: u32);
    }
}

pub mod cblas_z {
    use libc::{c_double, c_void};
    use crate::attribute::{
        Order,
        Transpose,
        Symmetry,
        Diagonal,
        Side,
    };

    pub use self::cblas_zgemm as gemm;
    pub use self::cblas_zhemm as hemm;
    pub use self::cblas_zher2k as her2k;
    pub use self::cblas_zherk as herk;
    pub use self::cblas_zsymm as symm;
    pub use self::cblas_zsyr2k as syr2k;
    pub use self::cblas_zsyrk as syrk;
    pub use self::cblas_ztrmm as trmm;
    pub use self::cblas_ztrsm as trsm;

    extern {
        pub fn cblas_zgemm(order: Order, trans_a: Transpose, trans_b: Transpose, m: u32, n: u32, k: u32, alpha: *const c_void, a: *const c_void,   lda: u32, b: *const c_void,   ldb: u32, beta: *const c_void, c: *mut c_void,   ldc: u32);
        pub fn cblas_zsymm(order: Order, side: Side, sym: Symmetry, m: u32, n: u32, alpha: *const c_void, a: *const c_void,   lda: u32, b: *const c_void,   ldb: u32, beta: *const c_void, c: *mut c_void,   ldc: u32);
        pub fn cblas_zhemm(order: Order, side: Side, sym: Symmetry, m: u32, n: u32, alpha: *const c_void, a: *const c_void,   lda: u32, b: *const c_void,   ldb: u32, beta: *const c_void, c: *mut c_void,   ldc: u32);
        pub fn cblas_ztrmm(order: Order, side: Side, sym: Symmetry, trans: Transpose, diag: Diagonal, m: u32, n: u32, alpha: *const c_void, a: *const c_void,   lda: u32, b: *mut c_void,   ldb: u32);
        pub fn cblas_ztrsm(order: Order, side: Side, sym: Symmetry, trans: Transpose, diag: Diagonal, m: u32, n: u32, alpha: *const c_void, a: *const c_void,   lda: u32, b: *mut c_void,   ldb: u32);
        pub fn cblas_zherk(order: Order, sym: Symmetry, Trans: Transpose, n: u32, k: u32, alpha: c_double, a: *const c_void, lda: u32, beta: c_double, c: *mut c_void, ldc: u32);
        pub fn cblas_zher2k(order: Order, sym: Symmetry, Trans: Transpose, n: u32, k: u32, alpha: *const c_void, a: *const c_void, lda: u32, b: *const c_void, ldb: u32, beta: c_double, c: *mut c_void, ldc: u32);
        pub fn cblas_zsyrk(order: Order, sym: Symmetry, Trans: Transpose, n: u32, k: u32, alpha: *const c_void, a: *const c_void,   lda: u32, beta: *const c_void, c: *mut c_void,   ldc: u32);
        pub fn cblas_zsyr2k(order: Order, sym: Symmetry, Trans: Transpose, n: u32, k: u32, alpha: *const c_void, a: *const c_void,   lda: u32, b: *const c_void,   ldb: u32, beta: *const c_void, c: *mut c_void,   ldc: u32);
    }
}
