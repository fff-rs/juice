//! Provides NN for a CUDA backend.

use ::operation::*;
use ::binary::*;
use ::plugin::*;
use co::backend::Backend;
use co::device::DeviceType;
use co::memory::MemoryType;
use co::tensor::{SharedTensor, ITensorDesc};
use co::plugin::Error as PluginError;
use co::frameworks::cuda::{Function, Module, Cuda};
use cudnn::*;

lazy_static! {
    static ref SIGMOID: Function = Function::from_isize(1);
    static ref CUDNN: Result<Cudnn, Error> = Cudnn::new();
}

pub trait ICudnnTensorDesc<T> {
    fn get_cudnn_desc(&self) -> Result<TensorDescriptor, PluginError>;
}

impl ICudnnTensorDesc<f32> for SharedTensor<f32> {
    fn get_cudnn_desc(&self) -> Result<TensorDescriptor, PluginError> {
        match TensorDescriptor::new(&self.desc().dims_i32().clone(), &self.desc().default_stride_i32().clone(), DataType::Float) {
            Ok(desc) => Ok(desc),
            Err(err) => {
                println!("{:?}", err);
                Err(PluginError::Plugin("Unable to create CuDNN TensorDescriptor."))
            }
        }
    }
}

impl ICudnnTensorDesc<f64> for SharedTensor<f64> {
    fn get_cudnn_desc(&self) -> Result<TensorDescriptor, PluginError> {
        match TensorDescriptor::new(&self.desc().dims_i32().clone(), &self.desc().default_stride_i32().clone(), DataType::Double) {
            Ok(desc) => Ok(desc),
            Err(err) => {
                println!("{:?}", err);
                Err(PluginError::Plugin("Unable to create CuDNN TensorDescriptor."))
            }
        }
    }
}

pub trait ICudnn {
    fn cudnn(&self) -> Cudnn {
        Cudnn::new().unwrap()
    }
}

impl ICudnn for Module {}

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
            fn compute(&self, x: &MemoryType, result: &mut MemoryType) -> Result<(), PluginError> {
                let cudnn = Cudnn::new().unwrap();
                let x = 2 * 2;
                unimplemented!();
                Ok(())
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
    impl_ops_relu_for!(f32, Backend<Cuda>);
    impl_ops_tanh_for!(f32, Backend<Cuda>);

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
    impl_ops_relu_for!(f64, Backend<Cuda>);
    impl_ops_tanh_for!(f64, Backend<Cuda>);

    fn binary(&self) -> &Self::B {
        self.binary()
    }

    fn device(&self) -> &DeviceType {
        self.device()
    }
}
