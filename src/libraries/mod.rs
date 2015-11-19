//! Exposes the specific library implementations.
//!
//! The purpose of a program is described at the [generic program module][program]. Collenchyma
//! comes shipped with the most basic Programs, which provide important low-level, backend-agnostic
//! numerical operations e.g. BLAS (Basic Linear Algebra Subprograms).
//!
//! Collenchyma can not provide all possible operations, but thanks to the functionality described
//! in the [program module][program], you can provide and run your own backend-agnostic operations,
//! too.
//!
//! [program]: ../program/index.html

pub mod blas;
