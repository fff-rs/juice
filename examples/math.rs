#[macro_use]
extern crate rblas as blas;
use blas::math::Mat;
use blas::{Matrix, Vector};
use blas::math::Marker::T;

fn main() {
    let x = vec![1.0, 2.0];
    let xr = &x as &Vector<_>;
    let i = mat![1.0, 0.0; 0.0, 1.0];
    let ir = &i as &Matrix<_>;

    assert!(xr + &x == 2.0 * xr);
    assert!(ir * xr == x);

    let dot = (xr ^ T) * xr;
    assert!(dot == 5.0);
}
