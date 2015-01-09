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
`Matrix` for types taht implement matrix-like characteristics. The `Vector`
trait is already implemented by `Vec` and `[]` types.

[Documentation](http://mikkyang.github.io/rust-blas/doc/rblas/index.html)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies.rblas]
git = "https://github.com/mikkyang/rust-blas.git"
```

and this to your crate root:
```rust
extern crate rblas;
```

## Example

```rust
extern crate rblas;

use rblas::Dot;

fn main() {
    let x = vec![1.0f32, -2.0, 3.0, 4.0];
    let y = vec![1.0f32, 1.0, 1.0, 1.0];

    let d: f32 = Dot::dot(&x, &y);
    assert_eq!(d, 6.0);
}
```
