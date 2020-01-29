// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

//! Bindings for vector functions.

pub mod cblas_s {
    use libc::{c_float, c_void};

    pub use self::cblas_sasum as asum;
    pub use self::cblas_saxpy as axpy;
    pub use self::cblas_scasum as casum;
    pub use self::cblas_scnrm2 as cnrm2;
    pub use self::cblas_scopy as copy;
    pub use self::cblas_sdot as dot;
    pub use self::cblas_sdsdot as dsdot;
    pub use self::cblas_snrm2 as nrm2;
    pub use self::cblas_srot as rot;
    pub use self::cblas_srotg as rotg;
    pub use self::cblas_srotm as rotm;
    pub use self::cblas_srotmg as rotmg;
    pub use self::cblas_sscal as scal;
    pub use self::cblas_sswap as swap;

    extern {
        pub fn cblas_scopy(n: u32, x: *const c_float,  inc_x: u32, y: *mut c_float,  inc_y: u32);
        pub fn cblas_saxpy(n: u32, alpha: c_float,       x: *const c_float,  inc_x: u32, y: *mut c_float,  inc_y: u32);
        pub fn cblas_sscal(n: u32, alpha: c_float,       x: *mut c_float,  inc_x: u32);
        pub fn cblas_sswap(n: u32, x: *mut c_float,  inc_x: u32, y: *mut c_float,  inc_y: u32);
        pub fn cblas_sdsdot(n: u32, alpha: c_float, x: *const c_float, inc_x: u32, y: *const c_float, inc_y: u32) -> c_float;
        pub fn cblas_sdot(n: u32, x: *const c_float,  inc_x: u32, y: *const c_float,  inc_y: u32) -> c_float;
        pub fn cblas_sasum(n: u32, x: *const c_float,  inc_x: u32) -> c_float;
        pub fn cblas_scasum(n: u32, x: *const c_void,   inc_x: u32) -> c_float;
        pub fn cblas_snrm2(n: u32, x: *const c_float,  inc_x: u32) -> c_float;
        pub fn cblas_scnrm2(n: u32, x: *const c_void,   inc_x: u32) -> c_float;
        pub fn cblas_srot(n: u32, x: *mut c_float,  inc_x: u32, y: *mut c_float,  inc_y: u32, c: c_float,  s: c_float);
        pub fn cblas_srotm(n: u32, x: *mut c_float,  inc_x: u32, y: *mut c_float,  inc_y: u32, p: *const c_float);
        pub fn cblas_srotg(a: *mut c_float,  b: *mut c_float,  c: *mut c_float,  s: *mut c_float);
        pub fn cblas_srotmg(d1: *mut c_float,  d2: *mut c_float,  b1: *mut c_float,  b2: c_float,  p: *mut c_float);
    }
}

pub mod cblas_d {
    use libc::{c_double, c_float, c_void};

    pub use self::cblas_dasum as asum;
    pub use self::cblas_daxpy as axpy;
    pub use self::cblas_dcopy as copy;
    pub use self::cblas_ddot as dot;
    pub use self::cblas_dnrm2 as nrm2;
    pub use self::cblas_drot as rot;
    pub use self::cblas_drotg as rotg;
    pub use self::cblas_drotm as rotm;
    pub use self::cblas_drotmg as rotmg;
    pub use self::cblas_dscal as scal;
    pub use self::cblas_dsdot as dsdot;
    pub use self::cblas_dswap as swap;
    pub use self::cblas_dzasum as zasum;
    pub use self::cblas_dznrm2 as znrm2;

    extern {
        pub fn cblas_dcopy(n: u32, x: *const c_double, inc_x: u32, y: *mut c_double, inc_y: u32);
        pub fn cblas_daxpy(n: u32, alpha: c_double,      x: *const c_double, inc_x: u32, y: *mut c_double, inc_y: u32);
        pub fn cblas_dscal (n: u32, alpha: c_double,      x: *mut c_double, inc_x: u32);
        pub fn cblas_dswap(n: u32, x: *mut c_double, inc_x: u32, y: *mut c_double, inc_y: u32);
        pub fn cblas_dsdot(n: u32, x: *const c_float,  inc_x: u32, y: *const c_float,  inc_y: u32) -> c_double;
        pub fn cblas_ddot (n: u32, x: *const c_double, inc_x: u32, y: *const c_double, inc_y: u32) -> c_double;
        pub fn cblas_dasum (n: u32, x: *const c_double, inc_x: u32) -> c_double;
        pub fn cblas_dzasum(n: u32, x: *const c_void,   inc_x: u32) -> c_double;
        pub fn cblas_dnrm2 (n: u32, x: *const c_double, inc_x: u32) -> c_double;
        pub fn cblas_dznrm2(n: u32, x: *const c_void,   inc_x: u32) -> c_double;
        pub fn cblas_drot(n: u32, x: *mut c_double, inc_x: u32, y: *mut c_double, inc_y: u32, c: c_double, s: c_double);
        pub fn cblas_drotm(n: u32, x: *mut c_double, inc_x: u32, y: *mut c_double, inc_y: u32, p: *const c_double);
        pub fn cblas_drotg(a: *mut c_double, b: *mut c_double, c: *mut c_double, s: *mut c_double);
        pub fn cblas_drotmg(
            d1: *mut c_double,
            d2: *mut c_double,
            b1: *mut c_double,
            b2: c_double,
            p: *mut c_double,
        );
    }
}

pub mod cblas_c {
    use libc::{c_float, c_void};

    pub use self::cblas_caxpy as axpy;
    pub use self::cblas_ccopy as copy;
    pub use self::cblas_cdotc_sub as dotc_sub;
    pub use self::cblas_cdotu_sub as dotu_sub;
    pub use self::cblas_cscal as scal;
    pub use self::cblas_csscal as sscal;
    pub use self::cblas_cswap as swap;

    extern {
        pub fn cblas_ccopy(n: u32, x: *const c_void,   inc_x: u32, y: *mut c_void,   inc_y: u32);
        pub fn cblas_caxpy(n: u32, alpha: *const c_void, x: *const c_void,   inc_x: u32, y: *mut c_void,   inc_y: u32);
        pub fn cblas_cscal (n: u32, alpha: *const c_void, x: *mut c_void,   inc_x: u32);
        pub fn cblas_csscal(n: u32, alpha: c_float,       x: *mut c_void,   inc_x: u32);
        pub fn cblas_cswap(n: u32, x: *mut c_void,   inc_x: u32, y: *mut c_void,   inc_y: u32);
        pub fn cblas_cdotu_sub(n: u32, x: *const c_void, inc_x: u32, y: *const c_void, inc_y: u32, dotu: *mut c_void);
        pub fn cblas_cdotc_sub(n: u32, x: *const c_void, inc_x: u32, y: *const c_void, inc_y: u32, dotc: *mut c_void);
    }
}

pub mod cblas_z {
    use libc::{c_double, c_void};

    pub use self::cblas_zaxpy as axpy;
    pub use self::cblas_zcopy as copy;
    pub use self::cblas_zdotc_sub as dotc_sub;
    pub use self::cblas_zdotu_sub as dotu_sub;
    pub use self::cblas_zdscal as dscal;
    pub use self::cblas_zscal as scal;
    pub use self::cblas_zswap as swap;

    extern {
        pub fn cblas_zcopy(n: u32, x: *const c_void,   inc_x: u32, y: *mut c_void,   inc_y: u32);
        pub fn cblas_zaxpy(n: u32, alpha: *const c_void, x: *const c_void,   inc_x: u32, y: *mut c_void,   inc_y: u32);
        pub fn cblas_zscal (n: u32, alpha: *const c_void, x: *mut c_void,   inc_x: u32);
        pub fn cblas_zdscal(n: u32, alpha: c_double,      x: *mut c_void,   inc_x: u32);
        pub fn cblas_zswap(n: u32, x: *mut c_void,   inc_x: u32, y: *mut c_void,   inc_y: u32);
        pub fn cblas_zdotu_sub(n: u32, x: *const c_void, inc_x: u32, y: *const c_void, inc_y: u32, dotu: *mut c_void);
        pub fn cblas_zdotc_sub(n: u32, x: *const c_void, inc_x: u32, y: *const c_void, inc_y: u32, dotc: *mut c_void);
    }
}

pub mod cblas_i {
    use libc::{c_double, c_float, c_void, size_t};

    pub use self::cblas_icamax as camax;
    pub use self::cblas_idamax as damax;
    pub use self::cblas_isamax as samax;
    pub use self::cblas_izamax as zamax;

    extern {
        pub fn cblas_isamax(n: u32, x: *const c_float,  inc_x: u32) -> size_t;
        pub fn cblas_idamax(n: u32, x: *const c_double, inc_x: u32) -> size_t;
        pub fn cblas_icamax(n: u32, x: *const c_void,   inc_x: u32) -> size_t;
        pub fn cblas_izamax(n: u32, x: *const c_void,   inc_x: u32) -> size_t;
    }
}
