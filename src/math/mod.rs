// Copyright 2015 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

pub mod vector;
pub mod matrix_vector;

pub enum Trans<A> {
    T(A),
    H(A),
}

pub enum Marker {
    T,
    H,
}
