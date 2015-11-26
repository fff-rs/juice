//! Provides BLAS for a Native backend.

use frameworks::native::{Function, Binary};
use libraries::blas::*;

impl IBlasBinary for Binary {
    type Dot = Function;

    fn dot(&self) -> Self::Dot {
        self.blas_dot
    }
}

impl IOperationDot for Function {
    fn compute(&self, a: i32) {
        println!("{}", format!("NATIVE"))
    }
}
