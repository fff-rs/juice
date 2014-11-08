// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

use std::raw::Repr;

pub mod ll;
pub mod ops;

pub trait Vector<T> {
    fn inc(&self) -> i32;
    fn len(&self) -> i32;
    fn as_ptr(&self) -> *const T;
    fn as_mut_ptr(&mut self) -> *mut T;
}

impl<T> Vector<T> for Vec<T> {
    #[inline]
    fn inc(&self) -> i32 { 1i32 }

    #[inline]
    fn len(&self) -> i32 {
        let l: Option<i32> = NumCast::from(self.len());
        match l {
            Some(l) => l,
            None => panic!(),
        }
    }

    #[inline]
    fn as_ptr(&self) -> *const T { self.as_slice().as_ptr() }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T { self.as_mut_slice().as_mut_ptr() }
}

impl<'a, T> Vector<T> for &'a [T] {
    #[inline]
    fn inc(&self) -> i32 { 1i32 }

    #[inline]
    fn len(&self) -> i32 {
        let l: Option<i32> = NumCast::from(self.len());
        match l {
            Some(l) => l,
            None => panic!(),
        }
    }

    #[inline]
    fn as_ptr(&self) -> *const T { self.repr().data }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T { self.repr().data as *mut T }
}
