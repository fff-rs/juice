//! Provides BLAS for a OpenCL backend.

use frameworks::OpenCL;
use frameworks::opencl::Kernel;
use frameworks::opencl::Program;
use binary::IBinary;
use libraries::blas::*;

impl IBlasBinary for Program {
    type Dot = Kernel;

    fn dot(&self) -> Self::Dot {
        self.blas_dot
    }
}

impl IOperationDot for Kernel {
    fn compute(&self, a: i32) {
        println!("{}", format!("OPENCL"))
    }
}
