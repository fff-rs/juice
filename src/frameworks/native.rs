//! Provides NN for a Native backend.

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unreachable_code)]

use ::plugin::*;
use co::device::DeviceType;
use co::backend::Backend;
use co::memory::MemoryType;
use co::frameworks::native::{Native, Function, Binary};
use co::plugin::Error;

struct ConvolutionConfig { dummy: i32 }

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
            unimplemented!();
            Ok(())
        }
        fn sigmoid_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }

        fn sigmoid_grad(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            x_diff: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }

        fn sigmoid_grad_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            x_diff: &::co::tensor::SharedTensor<$t>,
            result: &::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
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
            unimplemented!();
            Ok(())
        }

        fn relu_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }

        fn relu_grad(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            x_diff: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }

        fn relu_grad_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            x_diff: &::co::tensor::SharedTensor<$t>,
            result: &::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }
    );
}

macro_rules! impl_ops_tanh_for {
    ($t:ident, $b:ty) => (
        fn tanh(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }
        fn tanh_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }
        fn tanh_grad(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            x_diff: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
        }

        fn tanh_grad_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            x_diff: &::co::tensor::SharedTensor<$t>,
            result: &::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            unimplemented!();
            Ok(())
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
