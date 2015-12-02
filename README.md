# RBLAS

Rust bindings and wrappers for BLAS (Basic Linear Algebra Subprograms).

## Overview

RBLAS wraps each external call in a trait with the same name (but capitalized).
This trait contains a single static method, of the same name. These traits are
generic over the four main types of numbers BLAS supports: `f32`, `f64`,
`Complex32`, and `Complex64`.

For example the functions `cblas_saxpy`, `cblas_daxpy`, `cblas_caxypy`, and
`cblas_zaxpy` are called with the function `Axpy::axpy`.

Additionally, RBLAS introduces a few traits to shorten calls to these BLAS
functions: `Vector` for types that implement vector-like characteristics and
`Matrix` for types that implement matrix-like characteristics. The `Vector`
trait is already implemented by `Vec` and `[]` types.

[Documentation](http://mikkyang.github.io/rust-blas/doc/rblas/index.html)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rblas = "0.0.11"
```

and this to your crate root:
```rust
extern crate rblas;
```

By default, the library links with `blas` dynamically. To link to an alternate
implementation, like OpenBLAS, use the environment variable `CARGO_BLAS`. If
you've already built the bindings, you may need to clean and build again.

```
export CARGO_BLAS=openblas
```

## Example

```rust
extern crate rblas;

use rblas::Dot;

fn main() {
    let x = vec![1.0, -2.0, 3.0, 4.0];
    let y = [1.0, 1.0, 1.0, 1.0, 7.0];

    let d = Dot::dot(&x, &y[..x.len()]);
    assert_eq!(d, 6.0);
}
```

## Sugared Example

```rust
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
```
