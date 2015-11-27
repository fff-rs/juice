//! Provides BLAS for a Native backend.

use frameworks::native::{Function, Binary};
use memory::MemoryType;
use libraries::blas::*;

impl IBlasBinary for Binary {
    type Dot = Function;

    fn dot(&self) -> Self::Dot {
        self.blas_dot
    }
}

impl IOperationDot for Function {
    fn compute<T>(&self, x: &MemoryType, y: &MemoryType, result: &MemoryType) -> Result<(), Error> {
        match x {
            &MemoryType::Native(ref x) => {
                let x_slice = x.as_slice::<T>();
            },
            _ => ()
        }
        Ok(println!("{}", format!("NATIVE")))
    }
}
