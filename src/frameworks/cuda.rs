//! Provides NN for a CUDA backend.

use ::operation::*;
use ::binary::*;
use ::plugin::*;
use collenchyma::backend::Backend;
use collenchyma::device::DeviceType;
use collenchyma::memory::MemoryType;
use collenchyma::tensor::{SharedTensor, TensorDesc, ITensorDesc};
use collenchyma::plugin::Error as PluginError;
use collenchyma::frameworks::cuda::{Function, Module, Cuda};
use cudnn::*;
use std::mem::transmute;

lazy_static! {
    static ref SIGMOID: Function = Function::from_isize(1);
}

pub trait ICudnnTensorDesc : ITensorDesc {
    fn get_cudnn_desc(&self, data_type: DataType) -> Result<TensorDescriptor, PluginError> {
        match TensorDescriptor::new(&self.dims_i32(), &self.default_stride_i32(), data_type) {
            Ok(desc) => Ok(desc),
            Err(err) => {
                println!("{:?}", err);
                Err(PluginError::Plugin("Unable to create CuDNN TensorDescriptor."))
            }
        }
    }
}

impl ICudnnTensorDesc for TensorDesc {}

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

    fn sigmoid(&self, x: &mut SharedTensor<f32>, result: &mut SharedTensor<f32>) -> Result<(), ::collenchyma::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result.add_device(self.device()) { _ => () }
        let src_desc = try!(x.desc().get_cudnn_desc(DataType::Float));
        let src_data = try!(try!(x.get(self.device()).ok_or(PluginError::MissingMemoryForDevice("Unable to resolve memory for `x`")))
                .as_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive native memory for `x`.")))
                .id_c();
        let dest_desc = try!(result.desc().get_cudnn_desc(DataType::Float));
        let dest_data = try!(try!(result.get_mut(self.device()).ok_or(PluginError::MissingMemoryForDevice("Unable to resolve memory for `result`")))
                .as_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive native memory for `result`.")))
                .id_c();

        Ok(try!(match self.binary().cudnn().sigmoid_forward(
            &src_desc, unsafe { transmute::<u64, *const ::libc::c_void>(src_data) },
            &dest_desc, unsafe { transmute::<u64, *mut ::libc::c_void>(dest_data) },
            <ScalParams as IScalParamsDefault<f32>>::default()
        ) {
            Ok(_) => Ok(()),
            Err(err) => {
                println!("{:?}", err);
                Err(PluginError::Operation("Unable to execute CUDA cuDNN Activation Sigmoid Forward."))
            }
        }))
    }

    fn binary(&self) -> &Self::B {
        self.binary()
    }

    fn device(&self) -> &DeviceType {
        self.device()
    }
}

impl INn<f64> for Backend<Cuda> {
    type B = Module;

    fn sigmoid(&self, x: &mut SharedTensor<f64>, result: &mut SharedTensor<f64>) -> Result<(), ::collenchyma::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result.add_device(self.device()) { _ => () }
        let src_desc = try!(x.desc().get_cudnn_desc(DataType::Double));
        let src_data = try!(try!(x.get(self.device()).ok_or(PluginError::MissingMemoryForDevice("Unable to resolve memory for `x`")))
                .as_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive native memory for `x`.")))
                .id_c();
        let dest_desc = try!(result.desc().get_cudnn_desc(DataType::Double));
        let dest_data = try!(try!(result.get_mut(self.device()).ok_or(PluginError::MissingMemoryForDevice("Unable to resolve memory for `result`")))
                .as_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive native memory for `result`.")))
                .id_c();

        Ok(try!(match self.binary().cudnn().sigmoid_forward(
            &src_desc, unsafe { transmute::<u64, *const ::libc::c_void>(src_data) },
            &dest_desc, unsafe { transmute::<u64, *mut ::libc::c_void>(dest_data) },
            <ScalParams as IScalParamsDefault<f64>>::default()
        ) {
            Ok(_) => Ok(()),
            Err(err) => {
                println!("{:?}", err);
                Err(PluginError::Operation("Unable to execute CUDA cuDNN Activation Sigmoid Forward."))
            }
        }))
    }

    fn binary(&self) -> &Self::B {
        self.binary()
    }

    fn device(&self) -> &DeviceType {
        self.device()
    }
}
