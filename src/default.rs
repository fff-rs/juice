// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

use num_complex::Complex;

pub trait Default {
    fn one() -> Self;
    fn zero() -> Self;
    fn neg_one() -> Self;
}

macro_rules! default_impl(
    ($t:ty, $ov:expr, $zv:expr) => {
        impl Default for $t {
            #[inline]
            fn one() -> $t { $ov }
            #[inline]
            fn zero() -> $t { $zv }
            #[inline]
            fn neg_one() -> $t { -$ov }
        }

        impl Default for Complex<$t> {
            #[inline]
            fn one() -> Complex<$t> { Complex::new($ov, $zv) }
            #[inline]
            fn zero() -> Complex<$t> { Complex::new($zv, $zv) }
            #[inline]
            fn neg_one() -> Complex<$t> { Complex::new(-$ov, $zv) }
        }
    }
);

default_impl!(f32, 1f32, 0f32);
default_impl!(f64, 1f64, 0f64);
