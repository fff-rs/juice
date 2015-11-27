//! Provides BLAS for a OpenCL backend.

use frameworks::opencl::Kernel;
use frameworks::opencl::Program;
use memory::MemoryType;
use libraries::blas::*;

impl IBlasBinary for Program {
    type Dot = Kernel;

    fn dot(&self) -> Self::Dot {
        self.blas_dot
    }
}

impl IOperationDot for Kernel {
    fn compute<T>(&self, x: &MemoryType, y: &MemoryType, result: &mut MemoryType) -> Result<(), Error> {
        unimplemented!()
    }
}
