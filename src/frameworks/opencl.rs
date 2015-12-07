//! Provides BLAS for a OpenCL backend.

use ::operation::*;
use ::binary::*;
use ::library::*;
use collenchyma::backend::Backend;
use collenchyma::device::DeviceType;
use collenchyma::memory::MemoryType;
use collenchyma::plugin::Error;
use collenchyma::frameworks::opencl::{Kernel, Program, OpenCL};

impl IBlasBinary<f32> for Program {
    type Asum = Kernel;
    type Axpy = Kernel;
    type Copy = Kernel;
    type Dot = Kernel;
    type Nrm2 = Kernel;

    type Scale = Kernel;
    type Swap = Kernel;

    fn asum(&self) -> Self::Asum {
        unimplemented!()
    }

    fn axpy(&self) -> Self::Axpy {
        self.blas_axpy
    }

    fn copy(&self) -> Self::Copy {
        unimplemented!()
    }

    fn dot(&self) -> Self::Dot {
        self.blas_dot
    }

    fn nrm2(&self) -> Self::Nrm2 {
        unimplemented!()
    }

    fn scale(&self) -> Self::Scale {
        self.blas_scale
    }

    fn swap(&self) -> Self::Swap {
        unimplemented!()
    }
}

impl IOperationAsum<f32> for Kernel {
    fn compute(&self, x: &MemoryType, result: &mut MemoryType) -> Result<(), Error> {
        unimplemented!()
    }
}

impl IOperationAxpy<f32> for Kernel {
    fn compute(&self, a: &MemoryType, x: &MemoryType, y: &mut MemoryType) -> Result<(), Error> {
        unimplemented!()
    }
}

impl IOperationCopy<f32> for Kernel {
    fn compute(&self, x: &MemoryType, y: &mut MemoryType) -> Result<(), Error> {
        unimplemented!()
    }
}

impl IOperationDot<f32> for Kernel {
    fn compute(&self, x: &MemoryType, y: &MemoryType, result: &mut MemoryType) -> Result<(), Error> {
        unimplemented!()
    }
}

impl IOperationNrm2<f32> for Kernel {
    fn compute(&self, x: &MemoryType, result: &mut MemoryType) -> Result<(), Error> {
        unimplemented!()
    }
}

impl IOperationScale<f32> for Kernel {
    fn compute(&self, a: &MemoryType, x: &mut MemoryType) -> Result<(), Error> {
        unimplemented!()
    }
}

impl IOperationSwap<f32> for Kernel {
    fn compute(&self, x: &mut MemoryType, y: &mut MemoryType) -> Result<(), Error> {
        unimplemented!()
    }
}

impl IBlas<f32> for Backend<OpenCL> {
    type B = Program;

    fn binary(&self) -> &Self::B {
        self.binary()
    }

    fn device(&self) -> &DeviceType {
        self.device()
    }
}
