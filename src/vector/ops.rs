// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

extern crate num;

use self::num::complex::{Complex32, Complex64};
use pointer::CPtr;
use vector;
use vector::BlasVector;

pub trait Copy {
    fn copy(x: &BlasVector<Self>, y: &mut BlasVector<Self>);
}

macro_rules! copy_impl(
    ($t: ty, $copy_fn: ident) => (
        impl Copy for $t {
            fn copy(x: &BlasVector<$t>, y: &mut BlasVector<$t>) {
                unsafe {
                    vector::ll::$copy_fn(x.len(),
                        x.as_ptr().as_c_ptr(),  x.inc(),
                        y.as_mut_ptr().as_c_ptr(), y.inc());
                }
            }
        }
    );
)

copy_impl!(f32,       cblas_scopy)
copy_impl!(f64,       cblas_dcopy)
copy_impl!(Complex32, cblas_ccopy)
copy_impl!(Complex64, cblas_zcopy)
