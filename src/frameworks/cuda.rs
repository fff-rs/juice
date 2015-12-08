//! Provides NN for a CUDA backend.

use ::operation::*;
use ::binary::*;
use ::library::*;
use collenchyma::backend::Backend;
use collenchyma::device::DeviceType;
use collenchyma::memory::MemoryType;
use collenchyma::plugin::Error;
use collenchyma::frameworks::cuda::{Function, Module, Cuda};

lazy_static! {
    static ref SIGMOID: Function = Function::from_isize(1);
}

macro_rules! impl_binary(($($t: ident), +) => (
    $(
        impl INnBinary<$t> for Module {
            type Sigmoid = Function;

            fn sigmoid(&self) -> Self::Sigmoid {
                //lazy_static! {
                //    static ref SIGMOID: Function = Function::from_isize(1);
                //}
                Function::from_isize(1)
            }
        }
    )+
));

macro_rules! impl_sigmoid_for {
    ($t:ident, $b:ty) => (
        impl IOperationSigmoid<$t> for $b {
            fn compute(&self, x: &MemoryType, result: &mut MemoryType) -> Result<(), Error> {
                unimplemented!()
            }
        }
    );
}

macro_rules! impl_plugin_for {
    ($t:ident, $b:ty) => (
        impl_sigmoid_for!($t, $b);
    );
}

impl_binary!(f32, f64);
impl_plugin_for!(f32, Function);
impl_plugin_for!(f64, Function);

impl_plugin_for!(f32, Backend<Cuda>);
impl_plugin_for!(f64, Backend<Cuda>);

impl INn<f32> for Backend<Cuda> {
    type B = Module;

    impl_ops_sigmoid_for!(f32, Backend<Cuda>);

    fn binary(&self) -> &Self::B {
        self.binary()
    }

    fn device(&self) -> &DeviceType {
        self.device()
    }
}

impl INn<f64> for Backend<Cuda> {
    type B = Module;

    impl_ops_sigmoid_for!(f64, Backend<Cuda>);

    fn binary(&self) -> &Self::B {
        self.binary()
    }

    fn device(&self) -> &DeviceType {
        self.device()
    }
}
