//! Provides NN for a Native backend.

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unreachable_code)]

use ::plugin::*;
use co::prelude::*;
use co::Error;
use co::plugin::Error as PluginError;

#[macro_use]
pub mod helper;

impl_oconf_for_cc!(f32, f64);
impl_oconf_for_clrn!(f32, f64);
impl_oconf_for_pooling!(f32, f64);

impl NN<f32> for Backend<Native> {
    type CC = helper::ConvolutionConfig;
    type CLRN = helper::NormalizationConfig;
    type CPOOL = helper::PoolingConfig;

    fn init_nn() { }
    fn device(&self) -> &DeviceType { self.device() }
}

impl_ops_sigmoid_for!(f32, Backend<Native>);
impl_ops_relu_for!(f32, Backend<Native>);
impl_ops_tanh_for!(f32, Backend<Native>);
// impl_ops_convolution_for!(f32, Backend<Native>);
 impl_ops_softmax_for!(f32, Backend<Native>);
 impl_ops_log_softmax_for!(f32, Backend<Native>);
// impl_ops_lrn_for!(f32, Backend<Native>);
// impl_ops_pooling_for!(f32, Backend<Native>);

impl NN<f64> for Backend<Native> {
    type CC = helper::ConvolutionConfig;
    type CLRN = helper::NormalizationConfig;
    type CPOOL = helper::PoolingConfig;

    fn init_nn() { }
    fn device(&self) -> &DeviceType { self.device() }
}

impl_ops_sigmoid_for!(f64, Backend<Native>);
impl_ops_relu_for!(f64, Backend<Native>);
impl_ops_tanh_for!(f64, Backend<Native>);
// impl_ops_convolution_for!(f64, Backend<Native>);
 impl_ops_softmax_for!(f64, Backend<Native>);
 impl_ops_log_softmax_for!(f64, Backend<Native>);
// impl_ops_lrn_for!(f64, Backend<Native>);
// impl_ops_pooling_for!(f64, Backend<Native>);
