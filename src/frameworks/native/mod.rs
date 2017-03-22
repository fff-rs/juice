//! Provides NN for a Native backend.

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unreachable_code)]

use ::plugin::*;
use co::prelude::*;
use co::Error;
use co::plugin::Error as PluginError;
use co::plugin::numeric_helpers::Float;

#[macro_use]
pub mod helper;

// Those functions should be in helper.rs, but there is no point to make them
// public.
fn lens_eq<T>(xs: &[T], ys: &[T]) -> Result<(), Error> {
    if xs.len() != ys.len() {
        return Err(PluginError::Operation("Tensor dimension mismatch").into());
    }
    Ok(())
}


fn map1_inplace<T, F>(src: &mut [T], f: F) -> Result<(), Error>
    where T: Float,
          F: Fn(T) -> T {
    for i in 0..src.len() {
        src[i] = f(src[i]);
    }
    Ok(())
}

fn map2_inplace<T, F>(src1: &[T], src2: &mut [T], f: F) -> Result<(), Error>
    where T: Float,
          F: Fn(T, T) -> T {
    try!(lens_eq(src1, src2));
    for i in 0..src2.len() {
        src2[i] = f(src1[i], src2[i]);
    }
    Ok(())
}

fn map1<T, F>(src: &[T], dst: &mut [T], f: F) -> Result<(), Error>
    where T: Float,
          F: Fn(T) -> T {
    try!(lens_eq(dst, src));
    for i in 0..dst.len() {
        dst[i] = f(src[i]);
    }
    Ok(())
}

fn map2<T, F>(src1: &[T], src2: &[T], dst: &mut [T], f: F) -> Result<(), Error>
    where T: Float,
          F: Fn(T, T) -> T {
    try!(lens_eq(dst, src1));
    try!(lens_eq(dst, src2));
    for i in 0..dst.len() {
        dst[i] = f(src1[i], src2[i]);
    }
    Ok(())
}


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
