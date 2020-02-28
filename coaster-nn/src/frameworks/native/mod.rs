//! Provides NN for a Native backend.

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unreachable_code)]

use std::cmp::PartialOrd;
use std::fmt::Debug;
use std::ops::*;

use rand::{Rng, SeedableRng};
use rand_hc as hc128;

use crate::co::Error;
use crate::co::plugin::Error as PluginError;
use crate::co::plugin::numeric_helpers::Bounded;
use crate::co::plugin::numeric_helpers::Float;
use crate::co::prelude::*;
use crate::cudnn::{FilterDescriptor, TensorDescriptor};
use crate::plugin::*;

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
          F: Fn(T) -> T
{
    for i in 0..src.len() {
        src[i] = f(src[i]);
    }
    Ok(())
}

fn map2_inplace<T, F>(src1: &[T], src2: &mut [T], f: F) -> Result<(), Error>
    where T: Float,
          F: Fn(T, T) -> T
{
    lens_eq(src1, src2)?;
    for i in 0..src2.len() {
        src2[i] = f(src1[i], src2[i]);
    }
    Ok(())
}

fn map1<T, F>(src: &[T], dst: &mut [T], f: F) -> Result<(), Error>
    where T: Float,
          F: Fn(T) -> T
{
    lens_eq(dst, src)?;
    for i in 0..dst.len() {
        dst[i] = f(src[i]);
    }
    Ok(())
}

fn map2<T, F>(src1: &[T], src2: &[T], dst: &mut [T], f: F) -> Result<(), Error>
    where T: Float,
          F: Fn(T, T) -> T
{
    lens_eq(dst, src1)?;
    lens_eq(dst, src2)?;
    for i in 0..dst.len() {
        dst[i] = f(src1[i], src2[i]);
    }
    Ok(())
}


impl<T> NN<T> for Backend<Native>
    where T: Add<T, Output = T> + Mul<T, Output = T> + Default + Copy
{
    type CC = helper::ConvolutionConfig;
    type CLRN = helper::NormalizationConfig;
    type CPOOL = helper::PoolingConfig;
    // type CACTI = helper::ActivationConfig;
    type CDROP = helper::DropoutConfig;
    type CRNN = helper::RnnConfig;

    fn init_nn() {}
}

impl<'a, T> NNOperationConfig<T> for helper::ConvolutionConfig
    where T: Add<T, Output = T> + Mul<T, Output = T> + Default + Copy
{
}
impl<'a, T> ConvolutionConfig<T> for helper::ConvolutionConfig
    where T: Add<T, Output = T> + Mul<T, Output = T> + Default + Copy
{
}
impl<'a, T> RnnConfig<T> for helper::RnnConfig
where T: Add<T, Output = T> + Mul<T, Output = T> + Default + Copy
{
}
impl<T> NNOperationConfig<T> for helper::NormalizationConfig
    where T: Add<T, Output = T> + Mul<T, Output = T> + Default + Copy
{
}
impl<T> NNOperationConfig<T> for helper::PoolingConfig
    where T: Add<T, Output = T> + Mul<T, Output = T> + Default + Copy
{
}
// impl<T> NNOperationConfig<T> for helper::ActivationConfig
//     where T: Add<T, Output = T> + Mul<T, Output = T> + Default + Copy
// {
// }
impl<T> NNOperationConfig<T> for helper::DropoutConfig
    where T: Add<T, Output = T> + Mul<T, Output = T> + Default + Copy
{
}

impl<T> NNOperationConfig<T> for helper::RnnConfig
    where T: Add<T, Output = T> + Mul<T, Output = T> + Default + Copy
{
}

impl<T> Convolution<T> for Backend<Native>
    where T: Add<T, Output = T> + Mul<T, Output = T> + Default + Copy
{
    fn new_convolution_config(&self,
                              src: &SharedTensor<T>,
                              dest: &SharedTensor<T>,
                              filter: &SharedTensor<T>,
                              algo_fwd: ConvForwardAlgo,
                              algo_bwd_filter: ConvBackwardFilterAlgo,
                              algo_bwd_data: ConvBackwardDataAlgo,
                              stride: &[i32],
                              zero_padding: &[i32])
                              -> Result<Self::CC, Error> {
        // TODO: check dimensions of config
        match algo_fwd {
            ConvForwardAlgo::Auto |
            ConvForwardAlgo::ImplicitGEMM => {}
            _ => {
                return Err(Error::Plugin(PluginError::Plugin("Unimplemented.")));
            }
        }
        match algo_bwd_filter {
            ConvBackwardFilterAlgo::Auto |
            ConvBackwardFilterAlgo::ImplicitGEMM => {}
            _ => {
                return Err(Error::Plugin(PluginError::Plugin("Unimplemented.")));
            }
        }
        match algo_bwd_data {
            ConvBackwardDataAlgo::Auto |
            ConvBackwardDataAlgo::ImplicitGEMM => {}
            _ => {
                return Err(Error::Plugin(PluginError::Plugin("Unimplemented.")));
            }
        }

        Ok(helper::ConvolutionConfig {
               filter_shape: filter.desc().clone(),
               stride: stride.to_vec(),
               padding: zero_padding.to_vec(),
           })
    }

    fn convolution(&self,
                   filter: &SharedTensor<T>,
                   x: &SharedTensor<T>,
                   result: &mut SharedTensor<T>,
                   _workspace: &mut SharedTensor<u8>,
                   config: &Self::CC)
                   -> Result<(), Error> {
        let dev = self.device();

        let input_dim = x.desc();
        let input = x.read(dev)
            .unwrap()
            .as_slice::<T>();
        let input_stride = input_dim.default_stride();

        let output_dim = result.desc().clone();
        // this is ok, we only read parts we already wrote
        let output = result
            .write_only(dev)
            .unwrap()
            .as_mut_slice::<T>();

        let output_stride = output_dim.default_stride();
        {
            for o in output.iter_mut() {
                *o = Default::default();
            }
        }

        let filter_dim = filter.desc();
        let filter = filter
            .read(dev)
            .unwrap()
            .as_slice::<T>();
        let filter_stride = filter_dim.default_stride();

        // sanity check
        assert!(input_dim[0] == output_dim[0]);
        assert!(filter_dim[0] == output_dim[1]);
        assert!(input_dim[1] == filter_dim[1]);

        // TODO: specializations for spatial input

        // recursively sum up elementwise multiplication of the hyperplanes.
        fn filter_<T>(input: &[T],
                      input_stride: &[usize],
                      input_dim: &[usize],
                      input_offset: usize,
                      input_idx_base: &[usize],
                      filter: &[T],
                      filter_stride: &[usize],
                      filter_dim: &[usize],
                      filter_offset: usize,
                      padding: &[i32],
                      depth: usize,
                      depth_end: usize,
                      acc: Option<T>)
                      -> T
            where T: Add<T, Output = T> + Mul<T, Output = T> + Default + Copy
        {
            let mut acc = acc.unwrap_or_default();

            let p = padding[0] as usize;
            let input_idx_end = input_dim[0] + 2 * p;

            for filter_idx in 0..filter_dim[0] {
                let input_idx = input_idx_base[0] + filter_idx;
                let i_offset = input_offset + (input_idx - p) * input_stride[0];
                let f_offset = filter_offset + filter_idx * filter_stride[0];

                let v = if input_idx < p || input_idx + 1 > input_idx_end - p {
                    Default::default()
                } else if depth + 1 >= depth_end {
                    input[i_offset] * filter[f_offset]
                } else {
                    filter_(input,
                            &input_stride[1..],
                            &input_dim[1..],
                            i_offset,
                            &input_idx_base[1..],
                            filter,
                            &filter_stride[1..],
                            &filter_dim[1..],
                            f_offset,
                            &padding[1..],
                            depth + 1,
                            depth_end,
                            None)
                };
                acc = acc + v;
            }
            return acc;
        }

        // depth == 0 is the first level
        fn conv<T>(input: &[T],
                   input_stride: &[usize],
                   input_dim: &[usize],
                   top_input_offset: usize,
                   input_offset: usize,
                   input_idx_base: &mut [usize],
                   filter: &[T],
                   filter_stride: &[usize],
                   filter_dim: &[usize],
                   filter_offset: usize,
                   depth: usize,
                   padding: &[i32],
                   stride: &[i32],
                   output: &mut [T],
                   output_stride: &[usize],
                   output_dim: &[usize],
                   output_offset: usize)
            where T: Add<T, Output = T> + Mul<T, Output = T> + Default + Copy
        {
            let p = padding[depth] as usize;
            //let input_end = input_dim[depth] + 2 * p - (filter_dim[depth]);

            for output_idx in 0..output_dim[0] {
                let input_i = output_idx * stride[0] as usize;
                input_idx_base[depth] = input_i;
                let input_offset = input_offset + input_i * input_stride[depth];
                let output_offset = output_offset + output_idx * output_stride[0];

                if depth + 1 < input_dim.len() {
                    conv(input,
                         input_stride,
                         input_dim,
                         top_input_offset,
                         input_offset,
                         input_idx_base,
                         filter,
                         filter_stride,
                         filter_dim,
                         filter_offset,
                         depth + 1,
                         padding,
                         &stride[1..],
                         output,
                         &output_stride[1..],
                         &output_dim[1..],
                         output_offset);
                } else {
                    let v = filter_(input,
                                    input_stride,
                                    input_dim,
                                    top_input_offset,
                                    &input_idx_base[..],
                                    filter,
                                    filter_stride,
                                    filter_dim,
                                    filter_offset,
                                    padding,
                                    0,
                                    input_dim.len(),
                                    None);
                    output[output_offset] = output[output_offset] + v;
                }
            }
        }

        fn conv_k_d1<T>(_batch: usize,
                        input: &[T],
                        input_stride: &[usize],
                        input_dim: &[usize],
                        input_offset: usize,
                        input_idx_base: &mut [usize],
                        filter: &[T],
                        filter_stride: &[usize],
                        filter_dim: &[usize],
                        padding: &[i32],
                        stride: &[i32],
                        output: &mut [T],
                        output_stride: &[usize],
                        output_dim: &[usize],
                        output_offset: usize)
            where T: Add<T, Output = T> + Mul<T, Output = T> + Default + Copy
        {
            for k in 0..filter_dim[0] {
                let output_offset = output_offset + k * output_stride[0];
                let filter_offset = k * filter_stride[0];
                for d1 in 0..input_dim[0] {
                    let input_offset = input_offset + d1 * input_stride[0];
                    let filter_offset = filter_offset + d1 * filter_stride[1];

                    conv(input,
                         &input_stride[1..],
                         &input_dim[1..],
                         input_offset,
                         input_offset,
                         input_idx_base,
                         filter,
                         &filter_stride[2..],
                         &filter_dim[2..],
                         filter_offset,
                         0,
                         padding,
                         stride,
                         output,
                         &output_stride[1..],
                         &output_dim[1..],
                         output_offset);
                }
            }
        }

        let mut input_idx = Vec::new();
        input_idx.resize(input_dim.len() - 2, 0);
        let mut output_idx = Vec::new();
        output_idx.resize(output_dim.len(), 0);

        let batches = input_dim[0];
        for batch in 0..batches {
            let input_offset = batch * input_stride[0];
            let output_offset = batch * output_stride[0];

            conv_k_d1(batch,
                      input,
                      &input_stride[1..],
                      &input_dim[1..],
                      input_offset,
                      &mut input_idx[..],
                      filter,
                      &filter_stride[..],
                      &filter_dim[..],
                      &config.padding[..],
                      &config.stride[..],
                      output,
                      &output_stride[1..],
                      &output_dim[1..],
                      output_offset);
        }

        Ok(())
    }

    fn convolution_grad_filter(&self,
                               src_data: &SharedTensor<T>,
                               dest_diff: &SharedTensor<T>,
                               filter_diff: &mut SharedTensor<T>,
                               workspace: &mut SharedTensor<u8>,
                               config: &Self::CC)
                               -> Result<(), Error> {
        unimplemented!()
    }

    fn convolution_grad_data(&self,
                             filter: &SharedTensor<T>,
                             x_diff: &SharedTensor<T>,
                             result_diff: &mut SharedTensor<T>,
                             workspace: &mut SharedTensor<u8>,
                             config: &Self::CC)
                             -> Result<(), Error> {
        unimplemented!()
    }
}


impl<T> Pooling<T> for Backend<Native>
    where T: Add<T, Output = T> + Mul<T, Output = T> + Default + Copy + PartialOrd + Bounded
{
    fn new_pooling_config(&self,
                          window: &[i32],
                          stride: &[i32],
                          padding: &[i32])
                          -> Result<Self::CPOOL, Error> {
        Ok(helper::PoolingConfig {
               window: window.to_vec(),
               stride: stride.to_vec(),
               padding: padding.to_vec(),
           })
    }

    fn pooling_max(&self,
                   x: &SharedTensor<T>,
                   result: &mut SharedTensor<T>,
                   config: &Self::CPOOL)
                   -> Result<(), Error> {
        let dev = self.device();

        let input_dim = x.desc(); // [4, 4, 4, 4]
        let input = x.read(dev)
            .unwrap()
            .as_slice::<T>();
        let input_stride = input_dim.default_stride(); // [64, 16, 4, 1];

        let output_dim = result.desc().clone(); // [4,4,2,2]
        // this is ok, we only read parts we already wrote
        let output = result
            .write_only(dev)
            .unwrap()
            .as_mut_slice::<T>();
        let output_stride = output_dim.default_stride(); // [16, 4, 2, 1]
        {
            for o in output.iter_mut() {
                *o = Default::default();
            }
        }

        fn max_pooling_<T>(input: &[T],
                           input_stride: &[usize],
                           input_dim: &[usize],
                           input_offset: usize,
                           input_idx_base: &[usize],
                           window: &[i32],
                           padding: &[i32],
                           depth: usize,
                           depth_end: usize,
                           current_max: Option<T>)
                           -> T
            where T: Add<T, Output = T> + Mul<T, Output = T> + Default + Copy + PartialOrd + Bounded
        {
            let mut current_max = current_max.unwrap_or(T::min_value());

            let p = padding[0] as usize;
            let input_idx_end = input_dim[0] + 2 * p;

            for window_idx in 0..window[0] {
                let input_idx = input_idx_base[0] + window_idx as usize;

                let v = if input_idx < p || input_idx + 1 > input_idx_end - p {
                    T::min_value()
                } else {
                    let i_mem_offset = input_offset + (input_idx - p) * input_stride[0];
                    if depth + 1 >= depth_end {
                        input[i_mem_offset]
                    } else {
                        max_pooling_(input,
                                     &input_stride[1..],
                                     &input_dim[1..],
                                     i_mem_offset,
                                     &input_idx_base[1..],
                                     &window[1..],
                                     &padding[1..],
                                     depth + 1,
                                     depth_end,
                                     None)
                    }
                };
                // TODO: Handle NAN, inf and so on
                current_max = if current_max >= v {
                    current_max
                } else if current_max < v {
                    v
                } else {
                //TODO honour the configuration to pass on NaN or not, see cudnn API
                    panic!("NaN")
                };
            }
            current_max
        }

        fn recurse<T>(input: &[T],
                      input_stride: &[usize],
                      input_dim: &[usize],
                      top_input_offset: usize,
                      input_offset: usize,
                      input_idx_base: &mut [usize],
                      window: &[i32],
                      depth: usize,
                      stride: &[i32],
                      padding: &[i32],
                      output: &mut [T],
                      output_stride: &[usize],
                      output_dim: &[usize],
                      output_offset: usize)
            where T: Add<T, Output = T> + Mul<T, Output = T> + Default + Copy + PartialOrd + Bounded
        {
            let p = padding[depth] as usize; // 0
            let w = window[depth] as usize; // 2

            for output_idx in 0..output_dim[0] {
                let input_idx = output_idx * stride[0] as usize;
                input_idx_base[depth] = input_idx;
                // memory offset of linear input_idx
                let input_offset = input_offset + input_idx * input_stride[depth];
                let output_offset = output_offset + output_idx * output_stride[0];
                //println!("input_offset {} <- output_offset {}", input_offset, output_offset);

                if depth + 1 < input_dim.len() {
                    recurse(input,
                            input_stride,
                            input_dim,
                            top_input_offset,
                            input_offset,
                            input_idx_base,
                            window,
                            depth + 1,
                            &stride[1..],
                            padding,
                            output,
                            &output_stride[1..],
                            &output_dim[1..],
                            output_offset);
                } else {
                    let v = max_pooling_(input,
                                         input_stride,
                                         input_dim,
                                         top_input_offset,
                                         &input_idx_base[..],
                                         window,
                                         padding,
                                         0,
                                         input_dim.len(),
                                         None);
                    output[output_offset] = v;
                }
            }
        }


        let mut input_idx = Vec::new();
        input_idx.resize(input_dim.len() - 2, 0);
        let mut output_idx = Vec::new();
        output_idx.resize(output_dim.len(), 0);

        let window = &config.window[..];
        let stride = &config.stride[..];
        let padding = &config.padding[..];
        // do everything for each batch
        for batch in 0..input_dim[0] {
            // iterate over the batches!
            let input_offset = batch * input_stride[0];
            let output_offset = batch * output_stride[0];

            // iterate over the chanels
            for d1 in 0..input_dim[1] {
                let input_offset = input_offset + d1 * input_stride[1];
                let output_offset = output_offset + d1 * output_stride[1];
                // pass on the remaining dimensions (no batches, no channels, thus [2..]
                recurse(input,
                        &input_stride[2..],
                        &input_dim[2..],
                        input_offset,
                        input_offset,
                        &mut input_idx,
                        &window,
                        0,
                        &stride,
                        &padding,
                        output,
                        &output_stride[2..],
                        &output_dim[2..],
                        output_offset);
            }
        }

        Ok(())
    }

    // x, x_diff are known outputs of the forward propagation
    // result is the previous layer which derivate we want to know
    // FIXME verify
    fn pooling_max_grad(&self,
                        x: &SharedTensor<T>,
                        x_diff: &SharedTensor<T>,
                        result: &SharedTensor<T>,
                        result_diff: &mut SharedTensor<T>,
                        config: &Self::CPOOL)
                        -> Result<(), Error> {

        let dev = self.device();

        let input_dim = x.desc(); // []
        println!("x dims {:?}", input_dim);
        let input = x.read(dev)
            .unwrap()
            .as_slice::<T>();
        let input_stride = input_dim.default_stride(); // [];

        let x_diff_dim = x_diff.desc(); // []
        let x_diff = x_diff.read(dev).unwrap().as_slice::<T>();
        println!("x_diff dims {:?}", x_diff_dim);

        let output_dim = result_diff.desc().clone(); // []
        println!("result dims {:?}", result.desc());
        println!("result_diff dims {:?}", output_dim);

        // this is ok, we only read parts we already wrote
        let output = result_diff
            .write_only(dev)
            .unwrap()
            .as_mut_slice::<T>();
        let output_stride = output_dim.default_stride(); // []
        {
            for o in output.iter_mut() {
                *o = Default::default();
            }
        }

        fn max_pooling_<T>(input: &[T],
                           input_stride: &[usize],
                           input_dim: &[usize],
                           input_offset: usize,
                           input_idx_base: &[usize],
                           window: &[i32],
                           padding: &[i32],
                           depth: usize,
                           depth_end: usize,
                           current_max: Option<T>,
                           current_max_index: Option<usize>)
                           -> (T, usize)
            where T: Add<T, Output = T> + Mul<T, Output = T> + Default + Copy + PartialOrd + Bounded
        {
            let mut current_max = (current_max.unwrap_or(T::min_value()), current_max_index.unwrap_or(0usize));

            let p = padding[0] as usize;
            let input_idx_end = input_dim[0] + 2 * p;

            for window_idx in 0..window[0] {
                let input_idx = input_idx_base[0] + window_idx as usize;

                let (v, v_index) = if input_idx < p || input_idx + 1 > input_idx_end - p {
                    (T::min_value(), 0usize)
                } else {
                    let i_mem_offset = input_offset + (input_idx - p) * input_stride[0];
                    if depth + 1 >= depth_end {
                        (input[i_mem_offset], i_mem_offset)
                    } else {
                        max_pooling_(input,
                                     &input_stride[1..],
                                     &input_dim[1..],
                                     i_mem_offset,
                                     &input_idx_base[1..],
                                     &window[1..],
                                     &padding[1..],
                                     depth + 1,
                                     depth_end,
                                     None,
                                     None)
                    }
                };
                current_max = if current_max.0 >= v {
                    current_max
                } else if current_max.0 < v {
                    (v, v_index)
                } else {
                //TODO honour the configuration to pass on NaN or not, see cudnn API
                    panic!("NaN")
                };
            }
            current_max
        }

        fn recurse<T>(input: &[T],
                      input_stride: &[usize],
                      input_dim: &[usize],
                      top_input_offset: usize,
                      input_offset: usize,
                      input_idx_base: &mut [usize],
                      window: &[i32],
                      depth: usize,
                      stride: &[i32],
                      padding: &[i32],
                      output: &mut [T],
                      output_stride: &[usize],
                      output_dim: &[usize],
                      output_offset: usize,
                      dx: &[T])
            where T: Add<T, Output = T> + Mul<T, Output = T> + Default + Copy + PartialOrd + Bounded
        {
            let p = padding[depth] as usize; // 0
            let w = window[depth] as usize; // 2

            for output_idx in 0..output_dim[0] {
                let input_idx = output_idx * stride[0] as usize;
                input_idx_base[depth] = input_idx;
                // memory offset of linear input_idx
                let input_offset = input_offset + input_idx * input_stride[depth];
                let output_offset = output_offset + output_idx * output_stride[0];
                //println!("input_offset {} <- output_offset {}", input_offset, output_offset);

                if depth + 1 < input_dim.len() {
                    recurse(input,
                            input_stride,
                            input_dim,
                            top_input_offset,
                            input_offset,
                            input_idx_base,
                            window,
                            depth + 1,
                            &stride[1..],
                            padding,
                            output,
                            &output_stride[1..],
                            &output_dim[1..],
                            output_offset,
                            dx);
                } else {
                    let (val, index) = max_pooling_(input,
                                         input_stride,
                                         input_dim,
                                         top_input_offset,
                                         &input_idx_base[..],
                                         window,
                                         padding,
                                         0,
                                         input_dim.len(),
                                         None, None);
                    // if the stride is 1 and the size is i.e. multiple outputs of the forward propagation
                    // can map back to one input
                    // TODO sum up
                    output[index] = dx[0]; // FIXME we need a second index for this shit
                }
            }
        }


        let mut input_idx = Vec::new();
        input_idx.resize(input_dim.len() - 2, 0);
        let mut output_idx = Vec::new();
        output_idx.resize(output_dim.len(), 0);

        let window = &config.window[..];
        let stride = &config.stride[..];
        let padding = &config.padding[..];
        // do everything for each batch
        for batch in 0..input_dim[0] {
            // iterate over the batches!
            let input_offset = batch * input_stride[0];
            let output_offset = batch * output_stride[0];

            // iterate over the chanels
            for d1 in 0..input_dim[1] {
                let input_offset = input_offset + d1 * input_stride[1];
                let output_offset = output_offset + d1 * output_stride[1];
                // pass on the remaining dimensions (no batches, no channels, thus [2..]
                recurse(input,
                        &input_stride[2..],
                        &input_dim[2..],
                        input_offset,
                        input_offset,
                        &mut input_idx,
                        &window,
                        0,
                        &stride,
                        &padding,
                        output,
                        &output_stride[2..],
                        &output_dim[2..],
                        output_offset,
                        x_diff);
            }
        }
        Ok(())
    }

    fn pooling_avg(&self,
                   x: &SharedTensor<T>,
                   result: &mut SharedTensor<T>,
                   config: &Self::CPOOL)
                   -> Result<(), Error> {
        return Err(Error::Plugin(PluginError::Plugin("Unimplemented.")));
    }

    fn pooling_avg_grad(&self,
                        x: &SharedTensor<T>,
                        x_diff: &SharedTensor<T>,
                        result: &SharedTensor<T>,
                        result_diff: &mut SharedTensor<T>,
                        config: &Self::CPOOL)
                        -> Result<(), Error> {
        return Err(Error::Plugin(PluginError::Plugin("Unimplemented.")));
    }
}

impl<T> Rnn<T> for Backend<Native>
    where T: Float + Default + Copy + PartialOrd + Bounded {
    fn new_rnn_config(&self, src: &SharedTensor<T>, dropout_probability: Option<f32>, dropout_seed: Option<u64>, sequence_length: i32, network_mode: RnnNetworkMode, input_mode: RnnInputMode, direction_mode: DirectionMode, algorithm: RnnAlgorithm, hidden_size: i32, num_layers: i32, batch_size: i32) -> Result<Self::CRNN, Error> {
        // TODO: Implement Config to hold parameters regarding the RNN
        unimplemented!()
    }

    fn rnn_sequence_descriptors(&self,
                                src: &SharedTensor<T>,
                                sequence_length: i32,
                                input_size: i32,
                                hidden_size: i32,
                                batch_size: i32,
                                num_layers: i32)
                                -> Result<RnnSequenceDescriptors, Error> {
        unimplemented!()
    }

    fn generate_rnn_weight_description(
        &self,
        rnn_config: &Self::CRNN,
        sequence_length: i32,
        batch_size: i32,
        input_size: i32,
    ) -> Result<Vec<usize>, Error> {
        // This will end up being the tensor descriptor for the weights associated with the RNN pass
        unimplemented!()
    }

    fn rnn_forward(
        &self,
        src: &SharedTensor<T>,
        output: &mut SharedTensor<T>,
        rnn_config: &Self::CRNN,
        weight: &SharedTensor<T>,
        workspace: &mut SharedTensor<u8>,
    ) -> Result<(), Error> {
        unimplemented!()
    }

    fn rnn_backward_data(&self,
                         src: &SharedTensor<T>,
                         src_gradient: &mut SharedTensor<T>,
                         output: &SharedTensor<T>,
                         output_gradient: &SharedTensor<T>,
                         rnn_config: &Self::CRNN,
                         weight: &SharedTensor<T>,
                         workspace: &mut SharedTensor<u8>)
                         -> Result<(), Error> {
        unimplemented!()
    }

    fn rnn_backward_weights(&self,
                            src: &SharedTensor<T>,
                            output: &SharedTensor<T>,
                            filter: &mut SharedTensor<T>,
                            rnn_config: &Self::CRNN,
                            workspace: &mut SharedTensor<u8>)
                            -> Result<(), Error> { unimplemented!() }
}

impl<T> Dropout<T> for Backend<Native>
    where T: Float + Add<T, Output = T> + Mul<T, Output = T> + Default + Copy + PartialOrd + Bounded
{
    fn new_dropout_config(&self,
                      probability: f32,
                      seed: u64,
                      )
                      -> Result<Self::CDROP, Error> {
        Ok(helper::DropoutConfig{probability, seed})
    }

    // TODO this is supposed to be an in place operation
    fn dropout(&self,
           x: &SharedTensor<T>,
           result: &mut SharedTensor<T>,
           config: &Self::CDROP)
           -> Result<(), Error> {
        let dev = self.device();

        let input_dim = x.desc(); // [4, 4, 4, 4]
        let input = x.read(dev)
            .unwrap()
            .as_slice::<T>();

        let output_dim = result.desc().clone(); // [4,4,2,2]
        let output = result
            .write_only(dev)
            .unwrap()
            .as_mut_slice::<T>();

        output.clone_from_slice(input);


        let seed : [u8;8] = config.seed.to_le_bytes();
        let mut extrapolated_seed = [0u8; 32];
        extrapolated_seed[0..8].copy_from_slice(&seed[..]);
        let mut rng = hc128::Hc128Rng::from_seed(extrapolated_seed);

        for i in 0..output.len() {
            if rng.gen_range(0f32,1f32) >= config.probability {
                output[i] = input[i];
            } else {
                output[i] = T::zero();
            }
        }
        Ok(())
    }

    #[allow(unused_variables)]
    fn dropout_grad(&self,
                x: &SharedTensor<T>,
                x_diff: &SharedTensor<T>,
                result: &SharedTensor<T>,
                result_diff: &mut SharedTensor<T>,
                config: &Self::CDROP)
                -> Result<(), Error> {
        // TODO check if there is anything to do here?
        Ok(())
    }
}

// convolution is not needed here, it is well implemented without the macro madness
impl_ops_sigmoid_for!(f32, Backend<Native>);
impl_ops_relu_for!(f32, Backend<Native>);
impl_ops_tanh_for!(f32, Backend<Native>);
impl_ops_softmax_for!(f32, Backend<Native>);
impl_ops_log_softmax_for!(f32, Backend<Native>);
// impl_ops_lrn_for!(f32, Backend<Native>);

//impl NN<f64> for Backend<Native> {
//type CC = helper::ConvolutionConfig;
//type CLRN = helper::NormalizationConfig;
//type CPOOL = helper::PoolingConfig;

//fn init_nn() { }
//fn device(&self) -> &DeviceType { self.device() }
//}

impl_ops_sigmoid_for!(f64, Backend<Native>);
impl_ops_relu_for!(f64, Backend<Native>);
impl_ops_tanh_for!(f64, Backend<Native>);
impl_ops_softmax_for!(f64, Backend<Native>);
impl_ops_log_softmax_for!(f64, Backend<Native>);
// impl_ops_lrn_for!(f64, Backend<Native>);
