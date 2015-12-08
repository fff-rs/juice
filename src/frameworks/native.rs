//! Provides BLAS for a Native backend.

use ::operation::*;
use ::binary::*;
use ::library::*;
use collenchyma::device::DeviceType;
use collenchyma::backend::Backend;
use collenchyma::memory::MemoryType;
use collenchyma::frameworks::native::{Native, Function, Binary};
use collenchyma::plugin::Error;

macro_rules! impl_binary(($($t: ident), +) => (
    $(
        impl INnBinary<$t> for Binary {
            type Sigmoid = Function;

            fn sigmoid(&self) -> Self::Sigmoid {
                self.blas_asum
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

impl_plugin_for!(f32, Backend<Native>);
impl_plugin_for!(f64, Backend<Native>);

impl INn<f32> for Backend<Native> {
    type B = Binary;

    impl_ops_sigmoid_for!(f32, Backend<Native>);

    fn binary(&self) -> &Self::B {
        self.binary()
    }

    fn device(&self) -> &DeviceType {
        self.device()
    }
}

impl INn<f64> for Backend<Native> {
    type B = Binary;

    impl_ops_sigmoid_for!(f64, Backend<Native>);

    fn binary(&self) -> &Self::B {
        self.binary()
    }

    fn device(&self) -> &DeviceType {
        self.device()
    }
}
