//! Provides NN for a Native backend.

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unreachable_code)]

use ::plugin::*;
use co::plugin::numeric_helpers::Float;
use co::device::DeviceType;
use co::backend::Backend;
use co::memory::MemoryType;
use co::frameworks::native::{Native, Function, Binary};
use co::error::Error;
use co::plugin::Error as PluginError;

struct ConvolutionConfig { dummy: i32 }

fn write_to_memory<T: Iterator>(mem: &mut MemoryType, data: T) 
    where T::Item: Clone {
    let &mut MemoryType::Native(ref mut mem) = mem;
    let mut mem_buffer = mem.as_mut_slice::<T::Item>();
    for (index, datum) in data.enumerate() {
        mem_buffer[index] = datum;
    }
}

#[inline]
fn sigmoid<T: Float>(x: &T) -> T {
    let x : T = x.clone();
    (T::one()) / (T::one() + (-x).exp())
}

#[inline]
fn sigmoid_grad<T: Float>(x: &T, dx: &T) -> T {
    let x : T = x.clone();
    let dx : T = dx.clone();
    x * (T::one() -x) * dx
}

#[inline]
fn relu<T: Float>(x: &T) -> T {
    let x : T = x.clone();
    x.max(T::zero())
}

#[inline]
fn relu_grad<T: Float>(x: &T, dx: &T) -> T {
    let x : T = x.clone();
    let dx : T = dx.clone();
    if x > T::zero() {
        return dx
    }
    T::zero()
}

#[inline]
fn tanh<T: Float>(x: &T) -> T {
    let x : T = x.clone();
    x.tanh()
}

#[inline]
// d/dx tanh x = sech2 x = 1 + tanh2 x
fn tanh_grad<T: Float>(x: &T, dx: &T) -> T {
    let x : T = x.clone();
    let dx : T = dx.clone();
    (T::one() - x.powi(2)) * dx
}
 
#[macro_export]
macro_rules! impl_oconf_for_cc(($($t: ident), +) => (
    $(
        impl<'a> NNOperationConfig<$t> for ConvolutionConfig { }
    )+
));

struct NormalizationConfig { dummy: i32 }

#[macro_export]
macro_rules! impl_oconf_for_clrn(($($t: ident), +) => (
    $(
        impl NNOperationConfig<$t> for NormalizationConfig { }
    )+
));

struct PoolingConfig { dummy: i32 }

#[macro_export]
macro_rules! impl_oconf_for_pooling(($($t: ident), +) => (
    $(
        impl NNOperationConfig<$t> for PoolingConfig { }
    )+
));


macro_rules! impl_ops_sigmoid_for {
    ($t:ident, $b:ty) => (
        fn sigmoid(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result.add_device(self.device()) { _ => () }
            self.sigmoid_plain(x, result)
        }

        fn sigmoid_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            if let Some(input) = x.get(self.device()).unwrap().as_native() {
                let res = input.as_slice::<$t>().iter().map(sigmoid);
                write_to_memory(result.get_mut(self.device()).unwrap(), res);
                return Ok(());
            }
            Err(Error::Plugin(
                    PluginError::Operation("Unable to execute Native sigmoid Forward.")))
        }

        fn sigmoid_grad(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            x_diff: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match x_diff.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result_diff.add_device(self.device()) { _ => () }
            self.sigmoid_grad_plain(x, x_diff, result, result_diff)
        }

        fn sigmoid_grad_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            x_diff: &::co::tensor::SharedTensor<$t>,
            result: &::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {

            if let Some(sig_data) = x.get(self.device()).unwrap().as_native() {
                if let Some(sig_dx) = x.get(self.device()).unwrap().as_native() {
                    let res = sig_data.as_slice::<$t>().iter()
                        .zip(sig_dx.as_slice::<$t>().iter())
                        .map(|(t, dt)| sigmoid_grad(t, dt));
                    write_to_memory(result_diff.get_mut(self.device()).unwrap(), res);
                    return Ok(());
                }
            }
            Err(Error::Plugin(
                        PluginError::Operation("Unable to execute Native sigmoid grad Forward.")))
        }
    );
}

macro_rules! impl_ops_relu_for {
    ($t:ident, $b:ty) => (
        fn relu(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result.add_device(self.device()) { _ => () }
            self.relu_plain(x, result)
        }

        fn relu_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            if let Some(input) = x.get(self.device()).unwrap().as_native() {
                let res = input.as_slice::<$t>().iter().map(relu);
                write_to_memory(result.get_mut(self.device()).unwrap(), res);
                return Ok(());
            }
            Err(Error::Plugin(
                    PluginError::Operation("Unable to execute Native ReLU Forward.")))
        }

        fn relu_grad(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            x_diff: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match x_diff.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            self.relu_grad_plain(x, x_diff, result, result_diff)
        }

        fn relu_grad_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            x_diff: &::co::tensor::SharedTensor<$t>,
            result: &::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            if let Some(input) = x.get(self.device()).unwrap().as_native() {
                if let Some(dx) = x_diff.get(self.device()).unwrap().as_native() {
                    let res = input.as_slice::<$t>().iter()
                        .zip(dx.as_slice::<$t>().iter())
                        .map(|(x, dx)|relu_grad(x, dx));
                    write_to_memory(result_diff.get_mut(self.device()).unwrap(), res);
                    return Ok(());
                }
            }
            Err(Error::Plugin(
                    PluginError::Operation("Unable to execute Native ReLU grad Forward.")))

        }
    );
}

macro_rules! impl_ops_tanh_for {
    ($t:ident, $b:ty) => (
        #[inline]
        fn tanh(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result.add_device(self.device()) { _ => () }
            self.tanh_plain(x, result)
        }

        fn tanh_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            if let Some(input) = x.get(self.device()).unwrap().as_native() {
                let res = input.as_slice::<$t>().iter().map(tanh);
                write_to_memory(result.get_mut(self.device()).unwrap(), res);
                return Ok(());
            }
            Err(Error::Plugin(
                    PluginError::Operation("Unable to execute Native tanh Forward.")))
        }
        fn tanh_grad(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            x_diff: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match x_diff.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            self.tanh_grad_plain(x, x_diff, result, result_diff)
        }

        fn tanh_grad_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            x_diff: &::co::tensor::SharedTensor<$t>,
            result: &::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            if let Some(input) = x.get(self.device()).unwrap().as_native() {
                if let Some(dx) = x_diff.get(self.device()).unwrap().as_native() {
                    let res = input.as_slice::<$t>().iter()
                        .zip(dx.as_slice::<$t>().iter())
                        .map(|(x, dx)|tanh_grad(x, dx));
                    write_to_memory(result_diff.get_mut(self.device()).unwrap(), res);
                    return Ok(());
                }
            }
            Err(Error::Plugin(
                        PluginError::Operation("Unable to execute Native tanh_grad Forward.")))
        }
    );
}

macro_rules! impl_ops_convolution_for {
    ($t:ident, $b:ty) => (
        fn convolution(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CC
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }
        fn convolution_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CC
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }
        fn convolution_grad(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            x_diff: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CC
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }
        fn convolution_grad_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            x_diff: &::co::tensor::SharedTensor<$t>,
            result: &::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CC
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }
    );
}

macro_rules! impl_ops_softmax_for {
    ($t:ident, $b:ty) => (
        fn softmax(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }
        fn softmax_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }
        fn softmax_grad(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            x_diff: &mut ::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }
        fn softmax_grad_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            x_diff: &::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }
    );
}

macro_rules! impl_ops_lrn_for {
    ($t:ident, $b:ty) => (
        fn new_lrn_config(
        &self,
        n: u32,
        alpha: f64,
        beta: f64,
        k: f64
        ) -> Result<Self::CLRN, ::co::error::Error> {
            unimplemented!();
            Ok(NormalizationConfig{dummy: 7})
        }

        fn lrn(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CLRN
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }

        fn lrn_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CLRN
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }

        fn lrn_grad(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            x_diff: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CLRN
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }

        fn lrn_grad_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            x_diff: &::co::tensor::SharedTensor<$t>,
            result: &::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CLRN
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }
    );
}

macro_rules! impl_ops_pooling_for {
    ($t:ident, $b:ty) => (
        fn new_pooling_config(
            &self,
            window: &[i32],
            padding: &[i32],
            stride: &[i32]
        ) -> Result<Self::CPOOL, ::co::error::Error> {
            unimplemented!();
            Ok(PoolingConfig{dummy: 6})
        }

        fn pooling_max(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CPOOL
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }

        fn pooling_max_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CPOOL
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }
        #[allow(unused_variables)]
        fn pooling_max_grad(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            x_diff: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CPOOL
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }

        fn pooling_max_grad_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            x_diff: &::co::tensor::SharedTensor<$t>,
            result: &::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CPOOL
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }
    );
}

impl_oconf_for_cc!(f32, f64);
impl_oconf_for_clrn!(f32, f64);
impl_oconf_for_pooling!(f32, f64);

impl NN<f32> for Backend<Native> {
    type CC = ConvolutionConfig;
    type CLRN = NormalizationConfig;
    type CPOOL = PoolingConfig;

    fn init_nn() { }
    fn device(&self) -> &DeviceType { self.device() }

    impl_ops_sigmoid_for!(f32, Backend<Native>);
    impl_ops_relu_for!(f32, Backend<Native>);
    impl_ops_tanh_for!(f32, Backend<Native>);
    impl_ops_convolution_for!(f32, Backend<Native>);
    impl_ops_softmax_for!(f32, Backend<Native>);
    impl_ops_lrn_for!(f32, Backend<Native>);
    impl_ops_pooling_for!(f32, Backend<Native>);

    fn new_convolution_config(
        &self,
        src: &::co::tensor::SharedTensor<f32>,
        dest: &::co::tensor::SharedTensor<f32>,
        filter: &mut ::co::tensor::SharedTensor<f32>,
        stride: &[i32],
        zero_padding: &[i32]
    ) -> Result<Self::CC, ::co::error::Error> {
        unimplemented!();
        Ok(ConvolutionConfig{dummy: 5})
    }

}

impl NN<f64> for Backend<Native> {
    type CC = ConvolutionConfig;
    type CLRN = NormalizationConfig;
    type CPOOL = PoolingConfig;

    fn init_nn() { }
    fn device(&self) -> &DeviceType { self.device() }

    impl_ops_sigmoid_for!(f64, Backend<Native>);
    impl_ops_relu_for!(f64, Backend<Native>);
    impl_ops_tanh_for!(f64, Backend<Native>);
    impl_ops_convolution_for!(f64, Backend<Native>);
    impl_ops_softmax_for!(f64, Backend<Native>);
    impl_ops_lrn_for!(f64, Backend<Native>);
    impl_ops_pooling_for!(f64, Backend<Native>);

    fn new_convolution_config(
        &self,
        src: &::co::tensor::SharedTensor<f64>,
        dest: &::co::tensor::SharedTensor<f64>,
        filter: &mut ::co::tensor::SharedTensor<f64>,
        stride: &[i32],
        zero_padding: &[i32]
    ) -> Result<Self::CC, ::co::error::Error> {
        unimplemented!();
        Ok(ConvolutionConfig{dummy: 5})
    }
}
