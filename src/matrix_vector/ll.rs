// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

//! Bindings for matrix-vector functions.

pub mod cblas_s {
    use crate::attribute::{Diagonal, Order, Symmetry, Transpose};
    use libc::{c_float, c_int};

    pub use self::cblas_sgbmv as gbmv;
    pub use self::cblas_sgemv as gemv;
    pub use self::cblas_sger as ger;
    pub use self::cblas_ssbmv as sbmv;
    pub use self::cblas_sspmv as spmv;
    pub use self::cblas_sspr as spr;
    pub use self::cblas_sspr2 as spr2;
    pub use self::cblas_ssymv as symv;
    pub use self::cblas_ssyr as syr;
    pub use self::cblas_ssyr2 as syr2;
    pub use self::cblas_stbmv as tbmv;
    pub use self::cblas_stbsv as tbsv;
    pub use self::cblas_stpmv as tpmv;
    pub use self::cblas_stpsv as tpsv;
    pub use self::cblas_strmv as trmv;
    pub use self::cblas_strsv as trsv;

    extern "C" {
        pub fn cblas_sgemv(
            order: Order,
            trans: Transpose,
            m: c_int,
            n: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            x: *const c_float,
            inc_x: c_int,
            beta: c_float,
            y: *mut c_float,
            inc_y: c_int,
        );
        pub fn cblas_ssymv(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            x: *const c_float,
            inc_x: c_int,
            beta: c_float,
            y: *mut c_float,
            inc_y: c_int,
        );
        pub fn cblas_strmv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            a: *const c_float,
            lda: c_int,
            x: *mut c_float,
            inc_x: c_int,
        );
        pub fn cblas_strsv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            a: *const c_float,
            lda: c_int,
            x: *mut c_float,
            inc_x: c_int,
        );
        pub fn cblas_sger(
            order: Order,
            m: c_int,
            n: c_int,
            alpha: c_float,
            x: *const c_float,
            inc_x: c_int,
            y: *const c_float,
            inc_y: c_int,
            a: *mut c_float,
            lda: c_int,
        );
        pub fn cblas_ssyr(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: c_float,
            x: *const c_float,
            inc_x: c_int,
            a: *mut c_float,
            lda: c_int,
        );
        pub fn cblas_ssyr2(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: c_float,
            x: *const c_float,
            inc_x: c_int,
            y: *const c_float,
            inc_y: c_int,
            a: *mut c_float,
            lda: c_int,
        );
        pub fn cblas_sspmv(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: c_float,
            a: *const c_float,
            x: *const c_float,
            inc_x: c_int,
            beta: c_float,
            y: *mut c_float,
            inc_y: c_int,
        );
        pub fn cblas_sgbmv(
            order: Order,
            trans: Transpose,
            m: c_int,
            n: c_int,
            kl: c_int,
            ku: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            x: *const c_float,
            inc_x: c_int,
            beta: c_float,
            y: *mut c_float,
            inc_y: c_int,
        );
        pub fn cblas_ssbmv(
            order: Order,
            sym: Symmetry,
            n: c_int,
            k: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            x: *const c_float,
            inc_x: c_int,
            beta: c_float,
            y: *mut c_float,
            inc_y: c_int,
        );
        pub fn cblas_stbmv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            k: c_int,
            a: *const c_float,
            x: *mut c_float,
            inc_x: c_int,
        );
        pub fn cblas_stbsv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            k: c_int,
            a: *const c_float,
            x: *mut c_float,
            inc_x: c_int,
        );
        pub fn cblas_stpmv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            a: *const c_float,
            x: *mut c_float,
            inc_x: c_int,
        );
        pub fn cblas_stpsv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            a: *const c_float,
            x: *mut c_float,
            inc_x: c_int,
        );
        pub fn cblas_sspr(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: c_float,
            x: *const c_float,
            inc_x: c_int,
            a: *mut c_float,
        );
        pub fn cblas_sspr2(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: c_float,
            x: *const c_float,
            inc_x: c_int,
            y: *const c_float,
            inc_y: c_int,
            a: *mut c_float,
        );
    }
}

pub mod cblas_d {
    use crate::attribute::{Diagonal, Order, Symmetry, Transpose};
    use libc::{c_double, c_int};

    pub use self::cblas_dgbmv as gbmv;
    pub use self::cblas_dgemv as gemv;
    pub use self::cblas_dger as ger;
    pub use self::cblas_dsbmv as sbmv;
    pub use self::cblas_dspmv as spmv;
    pub use self::cblas_dspr as spr;
    pub use self::cblas_dspr2 as spr2;
    pub use self::cblas_dsymv as symv;
    pub use self::cblas_dsyr as syr;
    pub use self::cblas_dsyr2 as syr2;
    pub use self::cblas_dtbmv as tbmv;
    pub use self::cblas_dtbsv as tbsv;
    pub use self::cblas_dtpmv as tpmv;
    pub use self::cblas_dtpsv as tpsv;
    pub use self::cblas_dtrmv as trmv;
    pub use self::cblas_dtrsv as trsv;

    extern "C" {
        pub fn cblas_dgemv(
            order: Order,
            trans: Transpose,
            m: c_int,
            n: c_int,
            alpha: c_double,
            a: *const c_double,
            lda: c_int,
            x: *const c_double,
            inc_x: c_int,
            beta: c_double,
            y: *mut c_double,
            inc_y: c_int,
        );
        pub fn cblas_dsymv(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: c_double,
            a: *const c_double,
            lda: c_int,
            x: *const c_double,
            inc_x: c_int,
            beta: c_double,
            y: *mut c_double,
            inc_y: c_int,
        );
        pub fn cblas_dtrmv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            a: *const c_double,
            lda: c_int,
            x: *mut c_double,
            inc_x: c_int,
        );
        pub fn cblas_dtrsv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            a: *const c_double,
            lda: c_int,
            x: *mut c_double,
            inc_x: c_int,
        );
        pub fn cblas_dger(
            order: Order,
            m: c_int,
            n: c_int,
            alpha: c_double,
            x: *const c_double,
            inc_x: c_int,
            y: *const c_double,
            inc_y: c_int,
            a: *mut c_double,
            lda: c_int,
        );
        pub fn cblas_dsyr(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: c_double,
            x: *const c_double,
            inc_x: c_int,
            a: *mut c_double,
            lda: c_int,
        );
        pub fn cblas_dsyr2(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: c_double,
            x: *const c_double,
            inc_x: c_int,
            y: *const c_double,
            inc_y: c_int,
            a: *mut c_double,
            lda: c_int,
        );
        pub fn cblas_dspmv(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: c_double,
            a: *const c_double,
            x: *const c_double,
            inc_x: c_int,
            beta: c_double,
            y: *mut c_double,
            inc_y: c_int,
        );
        pub fn cblas_dgbmv(
            order: Order,
            trans: Transpose,
            m: c_int,
            n: c_int,
            kl: c_int,
            ku: c_int,
            alpha: c_double,
            a: *const c_double,
            lda: c_int,
            x: *const c_double,
            inc_x: c_int,
            beta: c_double,
            y: *mut c_double,
            inc_y: c_int,
        );
        pub fn cblas_dsbmv(
            order: Order,
            sym: Symmetry,
            n: c_int,
            k: c_int,
            alpha: c_double,
            a: *const c_double,
            lda: c_int,
            x: *const c_double,
            inc_x: c_int,
            beta: c_double,
            y: *mut c_double,
            inc_y: c_int,
        );
        pub fn cblas_dtbmv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            k: c_int,
            a: *const c_double,
            x: *mut c_double,
            inc_x: c_int,
        );
        pub fn cblas_dtbsv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            k: c_int,
            a: *const c_double,
            x: *mut c_double,
            inc_x: c_int,
        );
        pub fn cblas_dtpmv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            a: *const c_double,
            x: *mut c_double,
            inc_x: c_int,
        );
        pub fn cblas_dtpsv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            a: *const c_double,
            x: *mut c_double,
            inc_x: c_int,
        );
        pub fn cblas_dspr(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: c_double,
            x: *const c_double,
            inc_x: c_int,
            a: *mut c_double,
        );
        pub fn cblas_dspr2(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: c_double,
            x: *const c_double,
            inc_x: c_int,
            y: *const c_double,
            inc_y: c_int,
            a: *mut c_double,
        );
    }
}

pub mod cblas_c {
    use crate::attribute::{Diagonal, Order, Symmetry, Transpose};
    use libc::{c_float, c_int, c_void};

    pub use self::cblas_cgbmv as gbmv;
    pub use self::cblas_cgemv as gemv;
    pub use self::cblas_cgerc as gerc;
    pub use self::cblas_cgeru as geru;
    pub use self::cblas_chbmv as hbmv;
    pub use self::cblas_chemv as hemv;
    pub use self::cblas_cher as her;
    pub use self::cblas_cher2 as her2;
    pub use self::cblas_chpmv as hpmv;
    pub use self::cblas_chpr as hpr;
    pub use self::cblas_chpr2 as hpr2;
    pub use self::cblas_csymv as symv;
    pub use self::cblas_ctbmv as tbmv;
    pub use self::cblas_ctbsv as tbsv;
    pub use self::cblas_ctpmv as tpmv;
    pub use self::cblas_ctpsv as tpsv;
    pub use self::cblas_ctrmv as trmv;
    pub use self::cblas_ctrsv as trsv;

    extern "C" {
        pub fn cblas_cgemv(
            order: Order,
            trans: Transpose,
            m: c_int,
            n: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            x: *const c_void,
            inc_x: c_int,
            beta: *const c_void,
            y: *mut c_void,
            inc_y: c_int,
        );
        pub fn cblas_csymv(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            x: *const c_void,
            inc_x: c_int,
            beta: *const c_void,
            y: *mut c_void,
            inc_y: c_int,
        );
        pub fn cblas_chemv(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            x: *const c_void,
            inc_x: c_int,
            beta: *const c_void,
            y: *mut c_void,
            inc_y: c_int,
        );
        pub fn cblas_ctrmv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            a: *const c_void,
            lda: c_int,
            x: *mut c_void,
            inc_x: c_int,
        );
        pub fn cblas_ctrsv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            a: *const c_void,
            lda: c_int,
            x: *mut c_void,
            inc_x: c_int,
        );
        pub fn cblas_cgeru(
            order: Order,
            m: c_int,
            n: c_int,
            alpha: *const c_void,
            x: *const c_void,
            inc_x: c_int,
            y: *const c_void,
            inc_y: c_int,
            a: *mut c_void,
            lda: c_int,
        );
        pub fn cblas_cgerc(
            order: Order,
            m: c_int,
            n: c_int,
            alpha: *const c_void,
            x: *const c_void,
            inc_x: c_int,
            y: *const c_void,
            inc_y: c_int,
            a: *mut c_void,
            lda: c_int,
        );
        pub fn cblas_cher(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: c_float,
            x: *const c_void,
            inc_x: c_int,
            a: *mut c_void,
            lda: c_int,
        );
        pub fn cblas_cher2(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: *const c_void,
            x: *const c_void,
            inc_x: c_int,
            y: *const c_void,
            inc_y: c_int,
            a: *mut c_void,
            lda: c_int,
        );
        pub fn cblas_cgbmv(
            order: Order,
            trans: Transpose,
            m: c_int,
            n: c_int,
            kl: c_int,
            ku: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            x: *const c_void,
            inc_x: c_int,
            beta: *const c_void,
            y: *mut c_void,
            inc_y: c_int,
        );
        pub fn cblas_chbmv(
            order: Order,
            sym: Symmetry,
            n: c_int,
            k: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            x: *const c_void,
            inc_x: c_int,
            beta: *const c_void,
            y: *mut c_void,
            inc_y: c_int,
        );
        pub fn cblas_ctbmv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            k: c_int,
            a: *const c_void,
            x: *mut c_void,
            inc_x: c_int,
        );
        pub fn cblas_ctbsv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            k: c_int,
            a: *const c_void,
            x: *mut c_void,
            inc_x: c_int,
        );
        pub fn cblas_chpmv(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: *const c_void,
            a: *const c_void,
            x: *const c_void,
            inc_x: c_int,
            beta: *const c_void,
            y: *mut c_void,
            inc_y: c_int,
        );
        pub fn cblas_ctpmv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            a: *const c_void,
            x: *mut c_void,
            inc_x: c_int,
        );
        pub fn cblas_ctpsv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            a: *const c_void,
            x: *mut c_void,
            inc_x: c_int,
        );
        pub fn cblas_chpr(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: c_float,
            x: *const c_void,
            inc_x: c_int,
            a: *mut c_void,
        );
        pub fn cblas_chpr2(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: *const c_void,
            x: *const c_void,
            inc_x: c_int,
            y: *const c_void,
            inc_y: c_int,
            a: *mut c_void,
        );
    }
}

pub mod cblas_z {
    use crate::attribute::{Diagonal, Order, Symmetry, Transpose};
    use libc::{c_double, c_int, c_void};

    pub use self::cblas_zgbmv as gbmv;
    pub use self::cblas_zgemv as gemv;
    pub use self::cblas_zgerc as gerc;
    pub use self::cblas_zgeru as geru;
    pub use self::cblas_zhbmv as hbmv;
    pub use self::cblas_zhemv as hemv;
    pub use self::cblas_zher as her;
    pub use self::cblas_zher2 as her2;
    pub use self::cblas_zhpmv as hpmv;
    pub use self::cblas_zhpr as hpr;
    pub use self::cblas_zhpr2 as hpr2;
    pub use self::cblas_zsymv as symv;
    pub use self::cblas_ztbmv as tbmv;
    pub use self::cblas_ztbsv as tbsv;
    pub use self::cblas_ztpmv as tpmv;
    pub use self::cblas_ztpsv as tpsv;
    pub use self::cblas_ztrmv as trmv;
    pub use self::cblas_ztrsv as trsv;

    extern "C" {
        pub fn cblas_zgemv(
            order: Order,
            trans: Transpose,
            m: c_int,
            n: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            x: *const c_void,
            inc_x: c_int,
            beta: *const c_void,
            y: *mut c_void,
            inc_y: c_int,
        );
        pub fn cblas_zsymv(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            x: *const c_void,
            inc_x: c_int,
            beta: *const c_void,
            y: *mut c_void,
            inc_y: c_int,
        );
        pub fn cblas_zhemv(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            x: *const c_void,
            inc_x: c_int,
            beta: *const c_void,
            y: *mut c_void,
            inc_y: c_int,
        );
        pub fn cblas_ztrmv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            a: *const c_void,
            lda: c_int,
            x: *mut c_void,
            inc_x: c_int,
        );
        pub fn cblas_ztrsv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            a: *const c_void,
            lda: c_int,
            x: *mut c_void,
            inc_x: c_int,
        );
        pub fn cblas_zgeru(
            order: Order,
            m: c_int,
            n: c_int,
            alpha: *const c_void,
            x: *const c_void,
            inc_x: c_int,
            y: *const c_void,
            inc_y: c_int,
            a: *mut c_void,
            lda: c_int,
        );
        pub fn cblas_zgerc(
            order: Order,
            m: c_int,
            n: c_int,
            alpha: *const c_void,
            x: *const c_void,
            inc_x: c_int,
            y: *const c_void,
            inc_y: c_int,
            a: *mut c_void,
            lda: c_int,
        );
        pub fn cblas_zher(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: c_double,
            x: *const c_void,
            inc_x: c_int,
            a: *mut c_void,
            lda: c_int,
        );
        pub fn cblas_zher2(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: *const c_void,
            x: *const c_void,
            inc_x: c_int,
            y: *const c_void,
            inc_y: c_int,
            a: *mut c_void,
            lda: c_int,
        );
        pub fn cblas_zgbmv(
            order: Order,
            trans: Transpose,
            m: c_int,
            n: c_int,
            kl: c_int,
            ku: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            x: *const c_void,
            inc_x: c_int,
            beta: *const c_void,
            y: *mut c_void,
            inc_y: c_int,
        );
        pub fn cblas_zhbmv(
            order: Order,
            sym: Symmetry,
            n: c_int,
            k: c_int,
            alpha: *const c_void,
            a: *const c_void,
            lda: c_int,
            x: *const c_void,
            inc_x: c_int,
            beta: *const c_void,
            y: *mut c_void,
            inc_y: c_int,
        );
        pub fn cblas_ztbmv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            k: c_int,
            a: *const c_void,
            x: *mut c_void,
            inc_x: c_int,
        );
        pub fn cblas_ztbsv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            k: c_int,
            a: *const c_void,
            x: *mut c_void,
            inc_x: c_int,
        );
        pub fn cblas_zhpmv(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: *const c_void,
            a: *const c_void,
            x: *const c_void,
            inc_x: c_int,
            beta: *const c_void,
            y: *mut c_void,
            inc_y: c_int,
        );
        pub fn cblas_ztpmv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            a: *const c_void,
            x: *mut c_void,
            inc_x: c_int,
        );
        pub fn cblas_ztpsv(
            order: Order,
            sym: Symmetry,
            trans: Transpose,
            diag: Diagonal,
            n: c_int,
            a: *const c_void,
            x: *mut c_void,
            inc_x: c_int,
        );
        pub fn cblas_zhpr(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: c_double,
            x: *const c_void,
            inc_x: c_int,
            a: *mut c_void,
        );
        pub fn cblas_zhpr2(
            order: Order,
            sym: Symmetry,
            n: c_int,
            alpha: *const c_void,
            x: *const c_void,
            inc_x: c_int,
            y: *const c_void,
            inc_y: c_int,
            a: *mut c_void,
        );
    }
}
