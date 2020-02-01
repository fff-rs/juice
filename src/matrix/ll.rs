// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

//! Bindings for matrix functions.

pub mod cblas_s {
    use crate::attribute::{Diagonal, Order, Side, Symmetry, Transpose};
    use libc::{c_float, c_int};

    pub use self::cblas_sgemm as gemm;
    pub use self::cblas_ssymm as symm;
    pub use self::cblas_ssyr2k as syr2k;
    pub use self::cblas_ssyrk as syrk;
    pub use self::cblas_strmm as trmm;
    pub use self::cblas_strsm as strsm;

    extern "C" {
        pub fn cblas_sgemm(
            order: Order,
            trans_a: Transpose,
            trans_b: Transpose,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            b: *const c_float,
            ldb: c_int,
            beta: c_float,
            c: *mut c_float,
            ldc: c_int,
        );
        pub fn cblas_ssymm(
            order: Order,
            side: Side,
            sym: Symmetry,
            m: c_int,
            n: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            b: *const c_float,
            ldb: c_int,
            beta: c_float,
            c: *mut c_float,
            ldc: c_int,
        );
        pub fn cblas_strmm(
            order: Order,
            side: Side,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            m: c_int,
            n: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            b: *mut c_float,
            ldb: c_int,
        );
        pub fn cblas_strsm(
            order: Order,
            side: Side,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            m: c_int,
            n: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            b: *mut c_float,
            ldb: c_int,
        );
        pub fn cblas_ssyrk(
            order: Order,
            sym: Symmetry,
            Trans: Transpose,
            n: c_int,
            k: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            beta: c_float,
            c: *mut c_float,
            ldc: c_int,
        );
        pub fn cblas_ssyr2k(
            order: Order,
            sym: Symmetry,
            Trans: Transpose,
            n: c_int,
            k: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            b: *const c_float,
            ldb: c_int,
            beta: c_float,
            c: *mut c_float,
            ldc: c_int,
        );
    }
}

pub mod cblas_d {
    use crate::attribute::{Diagonal, Order, Side, Symmetry, Transpose};
    use libc::{c_double, c_int};

    pub use self::cblas_dgemm as gemm;
    pub use self::cblas_dsymm as symm;
    pub use self::cblas_dsyr2k as syr2k;
    pub use self::cblas_dsyrk as syrk;
    pub use self::cblas_dtrmm as trmm;
    pub use self::cblas_dtrsm as strsm;

    extern "C" {
        pub fn cblas_dgemm(
            order: Order,
            trans_a: Transpose,
            trans_b: Transpose,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: c_double,
            a: *const c_double,
            lda: c_int,
            b: *const c_double,
            ldb: c_int,
            beta: c_double,
            c: *mut c_double,
            ldc: c_int,
        );
        pub fn cblas_dsymm(
            order: Order,
            side: Side,
            sym: Symmetry,
            m: c_int,
            n: c_int,
            alpha: c_double,
            a: *const c_double,
            lda: c_int,
            b: *const c_double,
            ldb: c_int,
            beta: c_double,
            c: *mut c_double,
            ldc: c_int,
        );
        pub fn cblas_dtrmm(
            order: Order,
            side: Side,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            m: c_int,
            n: c_int,
            alpha: c_double,
            a: *const c_double,
            lda: c_int,
            b: *mut c_double,
            ldb: c_int,
        );
        pub fn cblas_dtrsm(
            order: Order,
            side: Side,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            m: c_int,
            n: c_int,
            alpha: c_double,
            a: *const c_double,
            lda: c_int,
            b: *mut c_double,
            ldb: c_int,
        );
        pub fn cblas_dsyrk(
            order: Order,
            sym: Symmetry,
            Trans: Transpose,
            n: c_int,
            k: c_int,
            alpha: c_double,
            a: *const c_double,
            lda: c_int,
            beta: c_double,
            c: *mut c_double,
            ldc: c_int,
        );
        pub fn cblas_dsyr2k(
            order: Order,
            sym: Symmetry,
            Trans: Transpose,
            n: c_int,
            k: c_int,
            alpha: c_double,
            a: *const c_double,
            lda: c_int,
            b: *const c_double,
            ldb: c_int,
            beta: c_double,
            c: *mut c_double,
            ldc: c_int,
        );
    }
}

pub mod cblas_c {
    use crate::attribute::{Diagonal, Order, Side, Symmetry, Transpose};
    use libc::{c_float, c_int, c_void};

    pub use self::cblas_cgemm as gemm;
    pub use self::cblas_chemm as hemm;
    pub use self::cblas_cher2k as her2k;
    pub use self::cblas_cherk as herk;
    pub use self::cblas_csymm as symm;
    pub use self::cblas_csyr2k as syr2k;
    pub use self::cblas_csyrk as syrk;
    pub use self::cblas_ctrmm as trmm;
    pub use self::cblas_ctrsm as trsm;

    extern "C" {
        pub fn cblas_cgemm(
            order: Order,
            trans_a: Transpose,
            trans_b: Transpose,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            b: *const c_void,
            ldb: c_int,
            beta: *const c_void,
            c: *mut c_void,
            ldc: c_int,
        );
        pub fn cblas_csymm(
            order: Order,
            side: Side,
            sym: Symmetry,
            m: c_int,
            n: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            b: *const c_void,
            ldb: c_int,
            beta: *const c_void,
            c: *mut c_void,
            ldc: c_int,
        );
        pub fn cblas_chemm(
            order: Order,
            side: Side,
            sym: Symmetry,
            m: c_int,
            n: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            b: *const c_void,
            ldb: c_int,
            beta: *const c_void,
            c: *mut c_void,
            ldc: c_int,
        );
        pub fn cblas_ctrmm(
            order: Order,
            side: Side,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            m: c_int,
            n: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            b: *mut c_void,
            ldb: c_int,
        );
        pub fn cblas_ctrsm(
            order: Order,
            side: Side,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            m: c_int,
            n: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            b: *mut c_void,
            ldb: c_int,
        );
        pub fn cblas_cherk(
            order: Order,
            sym: Symmetry,
            Trans: Transpose,
            n: c_int,
            k: c_int,
            alpha: c_float,
            a: *const c_void,
            lda: c_int,
            beta: c_float,
            c: *mut c_void,
            ldc: c_int,
        );
        pub fn cblas_cher2k(
            order: Order,
            sym: Symmetry,
            Trans: Transpose,
            n: c_int,
            k: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            b: *const c_void,
            ldb: c_int,
            beta: c_float,
            c: *mut c_void,
            ldc: c_int,
        );
        pub fn cblas_csyrk(
            order: Order,
            sym: Symmetry,
            Trans: Transpose,
            n: c_int,
            k: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            beta: *const c_void,
            c: *mut c_void,
            ldc: c_int,
        );
        pub fn cblas_csyr2k(
            order: Order,
            sym: Symmetry,
            Trans: Transpose,
            n: c_int,
            k: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            b: *const c_void,
            ldb: c_int,
            beta: *const c_void,
            c: *mut c_void,
            ldc: c_int,
        );
    }
}

pub mod cblas_z {
    use crate::attribute::{Diagonal, Order, Side, Symmetry, Transpose};
    use libc::{c_double, c_int, c_void};

    pub use self::cblas_zgemm as gemm;
    pub use self::cblas_zhemm as hemm;
    pub use self::cblas_zher2k as her2k;
    pub use self::cblas_zherk as herk;
    pub use self::cblas_zsymm as symm;
    pub use self::cblas_zsyr2k as syr2k;
    pub use self::cblas_zsyrk as syrk;
    pub use self::cblas_ztrmm as trmm;
    pub use self::cblas_ztrsm as trsm;

    extern "C" {
        pub fn cblas_zgemm(
            order: Order,
            trans_a: Transpose,
            trans_b: Transpose,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            b: *const c_void,
            ldb: c_int,
            beta: *const c_void,
            c: *mut c_void,
            ldc: c_int,
        );
        pub fn cblas_zsymm(
            order: Order,
            side: Side,
            sym: Symmetry,
            m: c_int,
            n: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            b: *const c_void,
            ldb: c_int,
            beta: *const c_void,
            c: *mut c_void,
            ldc: c_int,
        );
        pub fn cblas_zhemm(
            order: Order,
            side: Side,
            sym: Symmetry,
            m: c_int,
            n: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            b: *const c_void,
            ldb: c_int,
            beta: *const c_void,
            c: *mut c_void,
            ldc: c_int,
        );
        pub fn cblas_ztrmm(
            order: Order,
            side: Side,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            m: c_int,
            n: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            b: *mut c_void,
            ldb: c_int,
        );
        pub fn cblas_ztrsm(
            order: Order,
            side: Side,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            m: c_int,
            n: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            b: *mut c_void,
            ldb: c_int,
        );
        pub fn cblas_zherk(
            order: Order,
            sym: Symmetry,
            Trans: Transpose,
            n: c_int,
            k: c_int,
            alpha: c_double,
            a: *const c_void,
            lda: c_int,
            beta: c_double,
            c: *mut c_void,
            ldc: c_int,
        );
        pub fn cblas_zher2k(
            order: Order,
            sym: Symmetry,
            Trans: Transpose,
            n: c_int,
            k: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            b: *const c_void,
            ldb: c_int,
            beta: c_double,
            c: *mut c_void,
            ldc: c_int,
        );
        pub fn cblas_zsyrk(
            order: Order,
            sym: Symmetry,
            Trans: Transpose,
            n: c_int,
            k: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            beta: *const c_void,
            c: *mut c_void,
            ldc: c_int,
        );
        pub fn cblas_zsyr2k(
            order: Order,
            sym: Symmetry,
            Trans: Transpose,
            n: c_int,
            k: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            b: *const c_void,
            ldb: c_int,
            beta: *const c_void,
            c: *mut c_void,
            ldc: c_int,
        );
    }
}
