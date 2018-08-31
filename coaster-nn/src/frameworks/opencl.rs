//! Provides NN for a OpenCL backend.

use binary::*;
use co::Error;
use co::prelude::*;
use operation::*;
use plugin::*;

impl INnBinary<f32> for Program {
    type Sigmoid = Kernel;

    fn sigmoid(&self) -> Self::Sigmoid {
        unimplemented!()
    }
}

impl IOperationSigmoid<f32> for Kernel {
    fn compute(&self, x: &MemoryType, result: &mut MemoryType) -> Result<(), Error> {
        unimplemented!()
    }
}

impl INn<f32> for Backend<OpenCL> {
    type B = Program;

    fn binary(&self) -> &Self::B {
        self.binary()
    }

    fn device(&self) -> &DeviceType {
        self.device()
    }
}
