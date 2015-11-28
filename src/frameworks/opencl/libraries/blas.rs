//! Provides BLAS for a OpenCL backend.

use frameworks::opencl::Kernel;
use frameworks::opencl::Program;
use memory::MemoryType;
use libraries::blas::*;
use num::traits::Float;

impl IBlasBinary<f32> for Program {
    type Dot = Kernel;
    type Scale = Kernel;
    type Axpy = Kernel;

    fn dot(&self) -> Self::Dot {
        self.blas_dot
    }

    fn scale(&self) -> Self::Scale {
        self.blas_scale
    }

    fn axpy(&self) -> Self::Axpy {
        self.blas_axpy
    }
}

impl IOperationDot<f32> for Kernel {
    fn compute(&self, x: &MemoryType, y: &MemoryType, result: &mut MemoryType) -> Result<(), Error> {
        unimplemented!()
    }
}

impl IOperationScale<f32> for Kernel {
    fn compute(&self, a: &MemoryType, x: &mut MemoryType) -> Result<(), Error> {
        unimplemented!()
    }
}

impl IOperationAxpy<f32> for Kernel {
    fn compute(&self, a: &MemoryType, x: &MemoryType, y: &mut MemoryType) -> Result<(), Error> {
        unimplemented!()
    }
}
