//! Provides BLAS for a OpenCL backend.
// #![allow(unused_variables)]
// use ::operation::*;
// use ::binary::*;
// use ::library::*;
// use collenchyma::backend::Backend;
// use collenchyma::device::DeviceType;
// use collenchyma::memory::MemoryType;
// use collenchyma::plugin::Error;
// use collenchyma::frameworks::cuda::{Function, Module, Cuda};
//
// impl IBlasBinary<f32> for Module {
//     type Asum = Function;
//     type Axpy = Function;
//     type Copy = Function;
//     type Dot = Function;
//     type Nrm2 = Function;
//
//     type Scale = Function;
//     type Swap = Function;
//
//     fn asum(&self) -> Self::Asum {
//         unimplemented!()
//     }
//
//     fn axpy(&self) -> Self::Axpy {
//         self.blas_axpy
//     }
//
//     fn copy(&self) -> Self::Copy {
//         unimplemented!()
//     }
//
//     fn dot(&self) -> Self::Dot {
//         self.blas_dot
//     }
//
//     fn nrm2(&self) -> Self::Nrm2 {
//         unimplemented!()
//     }
//
//     fn scale(&self) -> Self::Scale {
//         self.blas_scale
//     }
//
//     fn swap(&self) -> Self::Swap {
//         unimplemented!()
//     }
// }
//
// impl IOperationAsum<f32> for Function {
//     fn compute(&self, x: &MemoryType, result: &mut MemoryType) -> Result<(), Error> {
//         unimplemented!()
//     }
// }
//
// impl IOperationAxpy<f32> for Function {
//     fn compute(&self, a: &MemoryType, x: &MemoryType, y: &mut MemoryType) -> Result<(), Error> {
//         unimplemented!()
//     }
// }
//
// impl IOperationCopy<f32> for Function {
//     fn compute(&self, x: &MemoryType, y: &mut MemoryType) -> Result<(), Error> {
//         unimplemented!()
//     }
// }
//
// impl IOperationDot<f32> for Function {
//     fn compute(&self, x: &MemoryType, y: &MemoryType, result: &mut MemoryType) -> Result<(), Error> {
//         unimplemented!()
//     }
// }
//
// impl IOperationNrm2<f32> for Function {
//     fn compute(&self, x: &MemoryType, result: &mut MemoryType) -> Result<(), Error> {
//         unimplemented!()
//     }
// }
//
// impl IOperationScale<f32> for Function {
//     fn compute(&self, a: &MemoryType, x: &mut MemoryType) -> Result<(), Error> {
//         unimplemented!()
//     }
// }
//
// impl IOperationSwap<f32> for Function {
//     fn compute(&self, x: &mut MemoryType, y: &mut MemoryType) -> Result<(), Error> {
//         unimplemented!()
//     }
// }
//
// impl IBlas<f32> for Backend<Cuda> {
//     type B = Module;
//
//     fn binary(&self) -> &Self::B {
//         self.binary()
//     }
//
//     fn device(&self) -> &DeviceType {
//         self.device()
//     }
// }
