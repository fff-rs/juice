//! Provides NN for a CUDA backend.
#![allow(missing_docs)]

use crate::co::Error;
use crate::co::plugin::Error as PluginError;
use crate::co::plugin::numeric_helpers::Float;
use crate::co::prelude::*;
use crate::cudnn::*;
pub use crate::cudnn::utils::{DataType, DataTypeInfo};
use crate::plugin::*;
use std::sync::{Arc, RwLock};

#[macro_use]
pub mod helper;

lazy_static! {
    static ref CUDNN: Cudnn = Cudnn::new().unwrap();
}

pub trait ICudnnDesc<T> {
    fn cudnn_tensor_desc(&self) -> Result<TensorDescriptor, PluginError>;
    /// Creates a TensorDescriptor similar to `cudnn_tensor_desc`,
    /// but will create a fitting 4D tensor if the actual tensor would be 1D-3D.
    fn cudnn_tensor_desc_softmax(&self) -> Result<TensorDescriptor, PluginError>;
    /// Creates a TensorDescriptor similar to `cudnn_tensor_desc`,
    /// but will create a fitting 3D tensor if the actual tensor would be 1D/2D.
    ///
    /// This should be used in operations where the shape doesn't really matter
    /// e.g. activation like ReLU.
    fn cudnn_tensor_desc_flat(&self) -> Result<TensorDescriptor, PluginError>;

    fn cudnn_filter_desc(&self) -> Result<FilterDescriptor, PluginError>;

    fn cudnn_convolution_desc(&self,
                              filter: &SharedTensor<T>)
                              -> Result<ConvolutionDescriptor, PluginError>;

    fn cudnn_rnn_desc( &self,
                       hidden_size: i32,
                       num_layers: i32,
                       dropout_desc: DropoutDescriptor,
                       input_mode: cudnnRNNInputMode_t,
                       direction: cudnnDirectionMode_t,
                       mode: cudnnRNNMode_t,
                       algorithm: cudnnRNNAlgo_t) -> Result<RnnDescriptor, PluginError>;
}

impl ConvForwardAlgo {
    /// Tries to return the matching cuDNN type for the enum value.
    fn as_cudnn(&self) -> Result<cudnnConvolutionFwdAlgo_t, Error> {
        use crate::ConvForwardAlgo::*;
        use crate::cudnn::cudnnConvolutionFwdAlgo_t::*;
        Ok(match *self {
            Auto => {
                return Err(Error::Plugin(PluginError::Plugin("Can't create cuDNN convolution forward algorithm from \
                 ConvForwardAlgo::Auto. Use `find_cudnn_algo` to find an algorithm.")))
            }
            GEMM => CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
            ImplicitGEMM => CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            ImplicitPrecompiledGEMM => CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
            FFT => CUDNN_CONVOLUTION_FWD_ALGO_FFT,
            FFTTiling => CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
            Direct => CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
            Winograd => CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
            WinogradNonFused => CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
        })
    }

    /// Returns the matching enum value for a cuDNN algo.
    fn from_cudnn(algo: &cudnnConvolutionFwdAlgo_t) -> ConvForwardAlgo {
        use crate::ConvForwardAlgo::*;
        use crate::cudnn::cudnnConvolutionFwdAlgo_t::*;
        match *algo {
            CUDNN_CONVOLUTION_FWD_ALGO_GEMM => GEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM => ImplicitGEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM => ImplicitPrecompiledGEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_FFT => FFT,
            CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING => FFTTiling,
            CUDNN_CONVOLUTION_FWD_ALGO_DIRECT => Direct,
            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD => Winograd,
            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED => WinogradNonFused,
            _ => unimplemented!(),
        }
    }

    /// Try to find best algorithm for a operation that uses the provided descriptors.
    fn find_cudnn_algo(&self,
                       filter_desc: &FilterDescriptor,
                       conv_desc: &ConvolutionDescriptor,
                       src_desc: &TensorDescriptor,
                       dest_desc: &TensorDescriptor)
                       -> Result<ConvForwardAlgo, Error> {
        if !self.is_auto() {
            return Ok(*self);
        }
        let algos = API::find_convolution_forward_algorithm(*CUDNN.id_c(),
                                                            *filter_desc.id_c(),
                                                            *conv_desc.id_c(),
                                                            *src_desc.id_c(),
                                                            *dest_desc.id_c())
            .unwrap();
        let algo = match algos.len() {
            0 => return Err(Error::Plugin(PluginError::Operation("Unable to find CUDA cuDNN convolution forward algorithm."))),
            _ => algos[0].algo,
        };
        Ok(ConvForwardAlgo::from_cudnn(&algo))
    }
}

impl ConvBackwardFilterAlgo {
    /// Tries to return the matching cuDNN type for the enum value.
    fn as_cudnn(&self) -> Result<cudnnConvolutionBwdFilterAlgo_t, Error> {
        use crate::ConvBackwardFilterAlgo::*;
        use crate::cudnn::cudnnConvolutionBwdFilterAlgo_t::*;
        Ok(match *self {
            Auto => {
                return Err(Error::Plugin(PluginError::Plugin("Can't create cuDNN convolution backward filter algorithm from \
                 ConvBackwardFilterAlgo::Auto. Use `find_cudnn_algo` to find an \
                 algorithm.")))
            }
            ImplicitGEMM => CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
            ImplicitGEMMSum => CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
            ImplicitPrecompiledGEMMSum => CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
            FFT => CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
            WinogradNonFused => CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
        })
    }

    /// Returns the matching enum value for a cuDNN algo.
    fn from_cudnn(algo: &cudnnConvolutionBwdFilterAlgo_t) -> ConvBackwardFilterAlgo {
        use crate::ConvBackwardFilterAlgo::*;
        use crate::cudnn::cudnnConvolutionBwdFilterAlgo_t::*;
        match *algo {
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 => ImplicitGEMMSum,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 => ImplicitGEMM,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT => FFT,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 => ImplicitPrecompiledGEMMSum,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED => WinogradNonFused,
            _ => unimplemented!(),
        }
    }

    /// Try to find best algorithm for a operation that uses the provided descriptors.
    fn find_cudnn_algo(&self,
                       filter_desc: &FilterDescriptor,
                       conv_desc: &ConvolutionDescriptor,
                       src_desc: &TensorDescriptor,
                       dest_desc: &TensorDescriptor)
                       -> Result<ConvBackwardFilterAlgo, Error> {
        if !self.is_auto() {
            return Ok(*self);
        }
        let algos = API::find_convolution_backward_filter_algorithm(*CUDNN.id_c(),
                                                                    *filter_desc.id_c(),
                                                                    *conv_desc.id_c(),
                                                                    *src_desc.id_c(),
                                                                    *dest_desc.id_c())
            .unwrap();
        let algo = match algos.len() {
            0 => return Err(Error::Plugin(PluginError::Operation("Unable to find CUDA cuDNN convolution backward filter algorithm."))),
            _ => algos[0].algo,
        };
        Ok(ConvBackwardFilterAlgo::from_cudnn(&algo))
    }
}

impl ConvBackwardDataAlgo {
    /// Tries to return the matching cuDNN type for the enum value.
    fn as_cudnn(&self) -> Result<cudnnConvolutionBwdDataAlgo_t, Error> {
        use crate::ConvBackwardDataAlgo::*;
        use crate::cudnn::cudnnConvolutionBwdDataAlgo_t::*;
        Ok(match *self {
            Auto => {
                return Err(Error::Plugin(PluginError::Plugin("Can't create cuDNN convolution backward data algorithm from \
                 ConvBackwardDataAlgo::Auto. Use `find_cudnn_algo` to find \
                 an algorithm.")))
            }
            ImplicitGEMM => CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
            ImplicitGEMMSum => CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
            FFT => CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
            FFTTiling => CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
            Winograd => CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
            WinogradNonFused => CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED,
        })
    }

    /// Returns the matching enum value for a cuDNN algo.
    fn from_cudnn(algo: &cudnnConvolutionBwdDataAlgo_t) -> ConvBackwardDataAlgo {
        use crate::ConvBackwardDataAlgo::*;
        use crate::cudnn::cudnnConvolutionBwdDataAlgo_t::*;
        match *algo {
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 => ImplicitGEMMSum,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 => ImplicitGEMM,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT => FFT,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING => FFTTiling,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD => Winograd,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED => WinogradNonFused,
            _ => unimplemented!(),
        }
    }

    /// Try to find best algorithm for a operation that uses the provided descriptors.
    fn find_cudnn_algo(&self,
                       filter_desc: &FilterDescriptor,
                       conv_desc: &ConvolutionDescriptor,
                       src_desc: &TensorDescriptor,
                       dest_desc: &TensorDescriptor)
                       -> Result<ConvBackwardDataAlgo, Error> {
        if !self.is_auto() {
            return Ok(*self);
        }
        let algos = API::find_convolution_backward_data_algorithm(*CUDNN.id_c(),
                                                                  *filter_desc.id_c(),
                                                                  *conv_desc.id_c(),
                                                                  *src_desc.id_c(),
                                                                  *dest_desc.id_c())
            .unwrap();

        let algo = match algos.len() {
            0 => return Err(Error::Plugin(PluginError::Operation("Unable to find CUDA cuDNN convolution backward data algorithm."))),
            _ => algos[0].algo,
        };
        Ok(ConvBackwardDataAlgo::from_cudnn(&algo))
    }
}

impl<T> ICudnnDesc<T> for SharedTensor<T>
    where T: Float + DataTypeInfo
{
    fn cudnn_tensor_desc(&self) -> Result<TensorDescriptor, PluginError> {
        match TensorDescriptor::new(&self.desc().dims_i32().clone(),
                                    &self.desc().default_stride_i32().clone(),
                                    <T as DataTypeInfo>::cudnn_data_type()) {
            Ok(desc) => Ok(desc),
            Err(_) => Err(PluginError::Plugin("Unable to create CuDNN TensorDescriptor.")),
        }
    }

    fn cudnn_tensor_desc_softmax(&self) -> Result<TensorDescriptor, PluginError> {
        let actual_desc = self.desc().clone();
        let override_desc = match actual_desc.len() {
            // not batched and single dimension softmax
            1 => vec![1, actual_desc[0], 1, 1],
            // batched and single dimension softmax
            2 => vec![actual_desc[0], actual_desc[1], 1, 1],
            // neither batched nor single dimension
            3 => vec![1, actual_desc[0], actual_desc[1], actual_desc[2]],
            _ => actual_desc,
        };
        match TensorDescriptor::new(&override_desc.dims_i32().clone(),
                                    &override_desc.default_stride_i32().clone(),
                                    <T as DataTypeInfo>::cudnn_data_type()) {
            Ok(desc) => Ok(desc),
            Err(_) => Err(PluginError::Plugin("Unable to create CuDNN TensorDescriptor.")),
        }
    }

    fn cudnn_tensor_desc_flat(&self) -> Result<TensorDescriptor, PluginError> {
        let actual_desc = self.desc().clone();
        let mut override_desc = match actual_desc.len() {
            1 => vec![1, 1],
            2 => vec![1],
            _ => vec![],
        };
        for dim in actual_desc {
            override_desc.push(dim);
        }
        match TensorDescriptor::new(&override_desc.dims_i32().clone(),
                                    &override_desc.default_stride_i32().clone(),
                                    <T as DataTypeInfo>::cudnn_data_type()) {
            Ok(desc) => Ok(desc),
            Err(_) => Err(PluginError::Plugin("Unable to create CuDNN TensorDescriptor.")),
        }
    }

    fn cudnn_filter_desc(&self) -> Result<FilterDescriptor, PluginError> {
        match FilterDescriptor::new(&self.desc().dims_i32().clone(),
                                    <T as DataTypeInfo>::cudnn_data_type()) {
            Ok(desc) => Ok(desc),
            Err(_) => Err(PluginError::Plugin("Unable to create CuDNN FilterDescriptor.")),
        }
    }

    //fn cudnn_tensor_desc_rnn(&self) -> Result<TensorDescriptor, PluginError> {
    //    let actual_desc : Vec<usize> = self.desc().clone();
    //    unimplemented!()
    //}

    fn cudnn_convolution_desc(&self,
                              filter: &SharedTensor<T>)
                              -> Result<ConvolutionDescriptor, PluginError> {
        match ConvolutionDescriptor::new(&self.desc().dims_i32().clone(),
                                         &filter.desc().default_stride_i32().clone(),
                                         <T as DataTypeInfo>::cudnn_data_type()) {
            Ok(desc) => Ok(desc),
            Err(_) => Err(PluginError::Plugin("Unable to create CuDNN ConvolutionDescriptor.")),
        }
    }

    fn cudnn_rnn_desc(
        &self,
        hidden_size: i32,
        num_layers: i32,
        dropout_desc: DropoutDescriptor,
        input_mode: cudnnRNNInputMode_t,
        direction: cudnnDirectionMode_t,
        mode: cudnnRNNMode_t,
        algorithm: cudnnRNNAlgo_t
    ) -> Result<RnnDescriptor, PluginError> {
        match RnnDescriptor::new(
            &CUDNN,
            hidden_size,
            num_layers,
            &dropout_desc,
            input_mode,
            direction,
            mode,
            algorithm,
            <T as DataTypeInfo>::cudnn_data_type(),
        ) {
            Ok(desc) => Ok(desc),
            Err(_) => Err(PluginError::Plugin("Unable to create CuDNN RNNDescriptor")),
        }
    }
}

impl<T> NN<T> for Backend<Cuda>
    where T: Float + DataTypeInfo
{
    type CC = utils::ConvolutionConfig;
    type CLRN = utils::NormalizationConfig;
    type CPOOL = utils::PoolingConfig;
    type CDROP = utils::DropoutConfig;
    type RC = utils::RnnConfig;

    fn init_nn() {
        let _ = CUDNN.id_c();
    }
}

impl<'a, T> NNOperationConfig<T> for utils::ConvolutionConfig where T: Float + DataTypeInfo {}
impl<T> NNOperationConfig<T> for utils::RnnConfig where T: Float + DataTypeInfo {}
impl<T> NNOperationConfig<T> for utils::NormalizationConfig where T: Float + DataTypeInfo {}
impl<T> NNOperationConfig<T> for utils::PoolingConfig where T: Float + DataTypeInfo {}
impl<T> NNOperationConfig<T> for utils::DropoutConfig where T: Float + DataTypeInfo {}

impl<T> Sigmoid<T> for Backend<Cuda>
    where T: Float + DataTypeInfo + Default
{
    fn sigmoid(&self,
               x: &SharedTensor<T>,
               result: &mut SharedTensor<T>)
               -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let r_desc = result.cudnn_tensor_desc_flat()?;
        let x_mem = read!(x, self);
        let r_mem = write_only!(result, self);
        match CUDNN.sigmoid_forward(&CUDNN.init_activation().unwrap(),

                                    &x.cudnn_tensor_desc_flat()?,
                                    trans!(x_mem),
                                    &r_desc,
                                    trans_mut!(r_mem),
                                    scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN Activation Sigmoid Forward."))),
        }
    }

    fn sigmoid_grad(&self,
                    x: &SharedTensor<T>,
                    x_diff: &SharedTensor<T>,
                    result: &SharedTensor<T>,
                    result_diff: &mut SharedTensor<T>)
                    -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let dr_desc = result_diff.cudnn_tensor_desc_flat()?;
        let x_mem = read!(x, self);
        let dx_mem = read!(x_diff, self);
        let r_mem = read!(result, self);
        let dr_mem = write_only!(result_diff, self);
        match CUDNN.sigmoid_backward(&CUDNN.init_activation().unwrap(),
                                     &x.cudnn_tensor_desc_flat()?,
                                     trans!(x_mem),
                                     &x_diff.cudnn_tensor_desc_flat()?,
                                     trans!(dx_mem),
                                     &result.cudnn_tensor_desc_flat()?,
                                     trans!(r_mem),
                                     &dr_desc,
                                     trans_mut!(dr_mem),
                                     scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN Activation Sigmoid Backward."))),
        }
    }
}

impl<T> ConvolutionConfig<T> for crate::cudnn::utils::ConvolutionConfig
    where T: Float + DataTypeInfo
{
    fn workspace_size(&self) -> usize {
        self.largest_workspace_size()
    }
}

impl<T> Convolution<T> for Backend<Cuda>
    where T: Float + DataTypeInfo
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
        let src_desc = src.cudnn_tensor_desc()?;
        let dest_desc = dest.cudnn_tensor_desc()?;
        let filter_desc = filter.cudnn_filter_desc()?;
        let conv_desc = crate::cudnn::ConvolutionDescriptor::new(zero_padding,
                                                                 stride,
                                                                 <T as DataTypeInfo>::cudnn_data_type())
            .unwrap();

        let useable_algo_fwd =
            algo_fwd.find_cudnn_algo(&filter_desc, &conv_desc, &src_desc, &dest_desc)?;
        let useable_algo_bwd_filter =
            algo_bwd_filter.find_cudnn_algo(&filter_desc, &conv_desc, &src_desc, &dest_desc)?;
        let useable_algo_bwd_data =
            algo_bwd_data.find_cudnn_algo(&filter_desc, &conv_desc, &src_desc, &dest_desc)?;

        let mut workspace_size_fwd =
            API::get_convolution_forward_workspace_size(*CUDNN.id_c(),
                                                        useable_algo_fwd.as_cudnn().unwrap(),
                                                        *filter_desc.id_c(),
                                                        *conv_desc.id_c(),
                                                        *src_desc.id_c(),
                                                        *dest_desc.id_c())
                .unwrap();
        let mut workspace_size_bwd_filter =
            API::get_convolution_backward_filter_workspace_size(*CUDNN.id_c(),
                                                                useable_algo_bwd_filter
                                                                    .as_cudnn()
                                                                    .unwrap(),
                                                                *filter_desc.id_c(),
                                                                *conv_desc.id_c(),
                                                                *src_desc.id_c(),
                                                                *dest_desc.id_c())
                .unwrap();
        let mut workspace_size_bwd_data =
            API::get_convolution_backward_data_workspace_size(*CUDNN.id_c(),
                                                              useable_algo_bwd_data
                                                                  .as_cudnn()
                                                                  .unwrap(),
                                                              *filter_desc.id_c(),
                                                              *conv_desc.id_c(),
                                                              *src_desc.id_c(),
                                                              *dest_desc.id_c())
                .unwrap();

        if workspace_size_fwd == 0 {
            workspace_size_fwd = 8;
        }
        if workspace_size_bwd_filter == 0 {
            workspace_size_bwd_filter = 8;
        }
        if workspace_size_bwd_data == 0 {
            workspace_size_bwd_data = 8;
        }

        Ok(crate::cudnn::utils::ConvolutionConfig::new(useable_algo_fwd.as_cudnn().unwrap(),
                                                  workspace_size_fwd,
                                                  useable_algo_bwd_filter.as_cudnn().unwrap(),
                                                  workspace_size_bwd_filter,
                                                  useable_algo_bwd_data.as_cudnn().unwrap(),
                                                  workspace_size_bwd_data,
                                                  conv_desc,
                                                  filter_desc))
    }
    fn convolution(&self,
                   filter: &SharedTensor<T>,
                   x: &SharedTensor<T>,
                   result: &mut SharedTensor<T>,
                   workspace: &mut SharedTensor<u8>,
                   config: &Self::CC)
                   -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();

        let r_desc = result.cudnn_tensor_desc()?;
        let f_mem = read!(filter, self);
        let x_mem = read!(x, self);
        let r_mem = write_only!(result, self);
        let w_mem = write_only!(workspace, self);

        match CUDNN.convolution_forward(config,
                                        trans_mut!(w_mem),
                                        trans!(f_mem),
                                        &x.cudnn_tensor_desc()?, // src_desc
                                        trans!(x_mem),
                                        &r_desc,
                                        trans_mut!(r_mem),
                                        scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN Activation convolution Forward."))),
        }
    }

    fn convolution_grad_filter(&self,
                               src_data: &SharedTensor<T>,
                               dest_diff: &SharedTensor<T>,
                               filter_diff: &mut SharedTensor<T>,
                               workspace: &mut SharedTensor<u8>,
                               config: &Self::CC)
                               -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let s_mem = read!(src_data, self);
        let dd_mem = read!(dest_diff, self);
        let df_mem = write_only!(filter_diff, self);
        let w_mem = write_only!(workspace, self);
        match CUDNN.convolution_backward_filter(config,
                                                trans_mut!(w_mem),
                                                &src_data.cudnn_tensor_desc()?,
                                                trans!(s_mem),
                                                &dest_diff.cudnn_tensor_desc()?,
                                                trans!(dd_mem),
                                                trans_mut!(df_mem),
                                                scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN Activation convolution Backward."))),
        }
    }

    fn convolution_grad_data(&self,
                             filter: &SharedTensor<T>,
                             x_diff: &SharedTensor<T>,
                             result_diff: &mut SharedTensor<T>,
                             workspace: &mut SharedTensor<u8>,
                             config: &Self::CC)
                             -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();

        let dr_desc = result_diff.cudnn_tensor_desc()?;
        let f_mem = read!(filter, self);
        let dx_mem = read!(x_diff, self);
        let dr_mem = write_only!(result_diff, self);
        let w_mem = write_only!(workspace, self);
        match CUDNN.convolution_backward_data(config,
                                              trans_mut!(w_mem),
                                              trans!(f_mem),
                                              &x_diff.cudnn_tensor_desc()?,
                                              trans!(dx_mem),
                                              &dr_desc,
                                              trans_mut!(dr_mem),
                                              scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Operation("Unable to execute CUDA cuDNN Activation convolution Backward."))),
        }
    }
}

impl<T> RnnConfig<T> for crate::cudnn::utils::RnnConfig where T: Float + DataTypeInfo
{
    fn workspace_size(&self) -> usize { self.largest_workspace_size() }
}

impl RnnInputMode {
    fn as_cudnn(&self) -> Result<cudnnRNNInputMode_t, Error> {
        Ok(match self {
            RnnInputMode::LinearInput => cudnnRNNInputMode_t::CUDNN_LINEAR_INPUT,
            RnnInputMode::SkipInput => cudnnRNNInputMode_t::CUDNN_SKIP_INPUT
        })
    }

    fn from_cudnn(input: cudnnRNNInputMode_t) -> Self {
        match input {
            cudnnRNNInputMode_t::CUDNN_LINEAR_INPUT => RnnInputMode::LinearInput,
            cudnnRNNInputMode_t::CUDNN_SKIP_INPUT => RnnInputMode::SkipInput
        }
    }
}

impl DirectionMode {
    fn as_cudnn(&self) -> Result<cudnnDirectionMode_t, Error> {
        Ok(match self {
            DirectionMode::BiDirectional => cudnnDirectionMode_t::CUDNN_BIDIRECTIONAL,
            DirectionMode::UniDirectional => cudnnDirectionMode_t::CUDNN_UNIDIRECTIONAL
        })
    }

    fn from_cudnn(direction: cudnnDirectionMode_t) -> Self {
        match direction {
            cudnnDirectionMode_t::CUDNN_BIDIRECTIONAL => DirectionMode::BiDirectional,
            cudnnDirectionMode_t::CUDNN_UNIDIRECTIONAL => DirectionMode::UniDirectional
        }
    }
}

impl RnnNetworkMode {
    fn as_cudnn(&self) -> Result<cudnnRNNMode_t, Error> {
        Ok(match self {
            RnnNetworkMode::ReLU => cudnnRNNMode_t::CUDNN_RNN_RELU,
            RnnNetworkMode::Tanh => cudnnRNNMode_t::CUDNN_RNN_TANH,
            RnnNetworkMode::LSTM => cudnnRNNMode_t::CUDNN_LSTM,
            RnnNetworkMode::GRU => cudnnRNNMode_t::CUDNN_GRU
        })
    }

    fn from_cudnn(network_mode: cudnnRNNMode_t) -> Self {
        match network_mode {
            cudnnRNNMode_t::CUDNN_RNN_RELU => RnnNetworkMode::ReLU,
            cudnnRNNMode_t::CUDNN_RNN_TANH => RnnNetworkMode::Tanh,
            cudnnRNNMode_t::CUDNN_LSTM => RnnNetworkMode::LSTM,
            cudnnRNNMode_t::CUDNN_GRU => RnnNetworkMode::GRU
        }
    }
}

impl RnnAlgorithm {
    fn as_cudnn(&self) -> Result<cudnnRNNAlgo_t, Error> {
        Ok(match self {
            RnnAlgorithm::PersistDynamic => cudnnRNNAlgo_t::CUDNN_RNN_ALGO_PERSIST_DYNAMIC,
            RnnAlgorithm::PersistStatic => cudnnRNNAlgo_t::CUDNN_RNN_ALGO_PERSIST_STATIC,
            RnnAlgorithm::Standard => cudnnRNNAlgo_t::CUDNN_RNN_ALGO_STANDARD
        })
    }

    fn from_cudnn(algorithm: cudnnRNNAlgo_t) -> Self {
        match algorithm {
            cudnnRNNAlgo_t::CUDNN_RNN_ALGO_PERSIST_DYNAMIC => RnnAlgorithm::PersistDynamic,
            cudnnRNNAlgo_t::CUDNN_RNN_ALGO_PERSIST_STATIC => RnnAlgorithm::PersistStatic,
            cudnnRNNAlgo_t::CUDNN_RNN_ALGO_STANDARD => RnnAlgorithm::Standard
        }
    }
}

impl MathType {
    fn as_cudnn(&self) -> Result<cudnnMathType_t, Error> {
        match self {
            MathType::Default => Ok(cudnnMathType_t::CUDNN_DEFAULT_MATH),
            MathType::TensorOPMath => Ok(cudnnMathType_t::CUDNN_TENSOR_OP_MATH),
            MathType::TensorOPMathAllowConversion => Err(Error::Plugin(PluginError::Plugin("TensorOPMathAllowConversion not yet supported.")))
        }
    }

    fn from_cudnn(math_type: cudnnMathType_t) -> MathType {
        match math_type {
            cudnnMathType_t::CUDNN_DEFAULT_MATH => MathType::Default,
            cudnnMathType_t::CUDNN_TENSOR_OP_MATH => MathType::TensorOPMath
        }
    }
}

#[derive(Debug)]
// All RNN Sequence Descriptors are generated on a single pass in CUDNN example code
// As such, defining them all in one function appears to be the simplest method of reproducing
// this work in Rust, but passing back a tuple is unwieldy as the tuple grows beyond 2 - 3 values.
pub struct RnnSequenceDescriptors {
    pub x_desc: Vec<TensorDescriptor>,
    y_desc: Vec<TensorDescriptor>,
    dx_desc: Vec<TensorDescriptor>,
    dy_desc: Vec<TensorDescriptor>,
    hx_desc: TensorDescriptor,
    cx_desc: TensorDescriptor,
    hy_desc: TensorDescriptor,
    cy_desc: TensorDescriptor,
    dhx_desc: TensorDescriptor,
    dcx_desc: TensorDescriptor,
    dhy_desc: TensorDescriptor,
    dcy_desc: TensorDescriptor,
}

impl<T> Rnn<T> for Backend<Cuda> where T: Float + DataTypeInfo {
    fn rnn_sequence_descriptors(&self,
                                src: &SharedTensor<T>,
                                sequence_length: i32,
                                input_size: i32,
                                batch_size: i32)
                                -> Result<RnnSequenceDescriptors, Error> {
        let mut x_desc: Vec<TensorDescriptor> = Vec::with_capacity(sequence_length as usize);
        let mut y_desc: Vec<TensorDescriptor> = Vec::with_capacity(sequence_length as usize);
        let mut dxdesc: Vec<TensorDescriptor> = Vec::with_capacity(sequence_length as usize);
        let mut dydesc: Vec<TensorDescriptor> = Vec::with_capacity(sequence_length as usize);
        let dim_a = vec![batch_size, input_size, 1];
        let stride_a = vec![dim_a[2] * dim_a[1], dim_a[2], 1];

        // FIXME: Ensure hidden_size*2 is used for bidirectional models
        let dim_b = vec![batch_size, input_size, 1];
        let stride_b = vec![dim_b[2] * dim_b[1], dim_b[2], 1];
        let data_type = <T as DataTypeInfo>::cudnn_data_type();
        let tensor_description_a = || {
            TensorDescriptor::new(
                &dim_a,
                &stride_a,
                data_type,
            ).unwrap()
        };
        // TODO: Move back to using closure when this finally passes.
        for _ in 0..sequence_length {
            x_desc.push(TensorDescriptor::new(
                &dim_a,
                &stride_a,
                data_type,
            ).unwrap());
            y_desc.push(TensorDescriptor::new(
                &dim_a,
                &stride_a,
                data_type,
            ).unwrap());
            dxdesc.push(TensorDescriptor::new(
                &dim_a,
                &stride_a,
                data_type,
            ).unwrap());
            dydesc.push(TensorDescriptor::new(
                &dim_a,
                &stride_a,
                data_type,
            ).unwrap());
        }
        let tensor_description_b = || {
            TensorDescriptor::new(
                &dim_b,
                &stride_b,
                <T as DataTypeInfo>::cudnn_data_type(),
            ).unwrap()
        };

        Ok(RnnSequenceDescriptors {
            x_desc,
            y_desc,
            dx_desc: dxdesc,
            dy_desc: dydesc,
            hx_desc: TensorDescriptor::new(
                &dim_b,
                &stride_b,
                <T as DataTypeInfo>::cudnn_data_type(),
            ).unwrap(),
            hy_desc: TensorDescriptor::new(
                &dim_b,
                &stride_b,
                <T as DataTypeInfo>::cudnn_data_type(),
            ).unwrap(),
            cx_desc: TensorDescriptor::new(
                &dim_b,
                &stride_b,
                <T as DataTypeInfo>::cudnn_data_type(),
            ).unwrap(),
            cy_desc: TensorDescriptor::new(
                &dim_b,
                &stride_b,
                <T as DataTypeInfo>::cudnn_data_type(),
            ).unwrap(),
            dhx_desc: TensorDescriptor::new(
                &dim_b,
                &stride_b,
                <T as DataTypeInfo>::cudnn_data_type(),
            ).unwrap(),
            dhy_desc: TensorDescriptor::new(
                &dim_b,
                &stride_b,
                <T as DataTypeInfo>::cudnn_data_type(),
            ).unwrap(),
            dcx_desc: TensorDescriptor::new(
                &dim_b,
                &stride_b,
                <T as DataTypeInfo>::cudnn_data_type(),
            ).unwrap(),
            dcy_desc: TensorDescriptor::new(
                &dim_b,
                &stride_b,
                <T as DataTypeInfo>::cudnn_data_type(),
            ).unwrap(),
        })
    }

    fn generate_rnn_weight_description(
        &self,
        rnn_config: &Self::RC,
        x_desc: &[TensorDescriptor]) -> Result<Vec<usize>, Error> {
        let weight_size: usize = match API::get_rnn_params_size(
            *CUDNN.id_c(),
            *rnn_config.rnn_desc().id_c(),
            *x_desc[0].id_c(),
            <T as DataTypeInfo>::cudnn_data_type()) {
            Ok(size) => Ok(size),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to get CudNN Rnn Params Size."))),
        }?;
        //let linLayerMat = match API::get_rnn_lin_layer_matrix_params();
        dbg!(weight_size);
        // This is taken from https://github.com/Hardware-Alchemy/cuDNN-sample/blob/master/cudnn_samples_v7/RNN/RNN_example.cu#L302-L305
        // Where [weight_size, 1, 1] is used to define the filter for wDesc and dwDesc.
        let dim_w: Vec<usize> = vec![weight_size, 1, 1];
        Ok(dim_w)
    }

    fn new_rnn_config(
        &self,
        src: &SharedTensor<T>,
        dropout_probability: Option<f32>,
        dropout_seed: Option<u64>,
        sequence_length: i32,
        network_mode: RnnNetworkMode,
        input_mode: RnnInputMode,
        direction_mode: DirectionMode,
        algorithm: RnnAlgorithm,
        hidden_size: i32,
        num_layers: i32,
        batch_size: i32,
    ) -> Result<Self::RC, Error> {
        let input_mode = input_mode.as_cudnn()?;
        let direction_mode = direction_mode.as_cudnn()?;
        let network_mode = network_mode.as_cudnn()?;
        let algorithm = algorithm.as_cudnn()?;

        let drop_desc = match CUDNN.init_dropout(
            dropout_probability.unwrap_or(0.5),
            dropout_seed.unwrap_or(0),
        ) {
            Ok(dropout_object) => Ok(dropout_object),
            Err(E) => Err(Error::Plugin(PluginError::Plugin("Unable to create Dropout Layer")))
        }?;

        let x_desc = self.rnn_sequence_descriptors(
            src,
            sequence_length,
            hidden_size,
            batch_size,
        )?.x_desc;

        let rnn_desc = match RnnDescriptor::new(
            &CUDNN,
            hidden_size,
            num_layers,
            drop_desc.dropout_desc(),
            input_mode,
            direction_mode,
            network_mode,
            algorithm,
            <T as DataTypeInfo>::cudnn_data_type(),
        ) {
            Ok(desc) => desc,
            Err(e) => panic!("Error {:?}", e)
        };

        match CUDNN.init_rnn(
            &x_desc,
            rnn_desc,
            hidden_size,
            batch_size,
            sequence_length,
            num_layers,
            drop_desc.dropout_desc(),
            input_mode,
            direction_mode,
            network_mode,
            algorithm,
            <T as DataTypeInfo>::cudnn_data_type(),
            MathType::TensorOPMath.as_cudnn()?,
        ) {
            Ok(rnn_config) => Ok(rnn_config),
            Err(e) => panic!("Error {:?}", e)
        }
    }

    /// Train and Output a RNN Network
    fn rnn_forward(
        &self,
        src: &SharedTensor<T>,
        output: &mut SharedTensor<T>,
        rnn_config: &Self::RC,
        weight: &SharedTensor<T>,
        workspace: &mut SharedTensor<u8>
    ) -> Result<(), Error> {
        let src_dimensions = src.desc().clone();
        let sequence_descriptors = self.rnn_sequence_descriptors(
            src,
            *rnn_config.sequence_length(),
            rnn_config.hidden_size,
            src_dimensions[0] as i32,
        )?;
        let weight_desc = weight.cudnn_filter_desc()?;
        let reserve = Some(Arc::new(RwLock::new(
            SharedTensor::<u8>::new(&[rnn_config.training_reserve_size()]))));
        let reserve_space = &mut reserve.as_ref().unwrap().write().unwrap();

        let src_mem = read!(src, self);
        let weight_mem = read!(weight, self);
        let output_mem = write_only!(output, self);
        let workspace_mem = write_only!(workspace, self);
        let reserve_mem = write_only!(reserve_space, self);

        match CUDNN.rnn_forward::<f32>(
            rnn_config,
            sequence_descriptors.x_desc,
            trans!(src_mem),
            sequence_descriptors.y_desc,
            trans_mut!(output_mem),
            &sequence_descriptors.hx_desc,
            std::ptr::null(),
            &sequence_descriptors.cx_desc,
            std::ptr::null(),
            &weight_desc,
            trans!(weight_mem),
            &sequence_descriptors.hy_desc,
            std::ptr::null_mut(),
            &sequence_descriptors.cy_desc,
            std::ptr::null_mut(),
            trans_mut!(workspace_mem),
            trans_mut!(reserve_mem),
        ) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to perform RNN Forward")))
        }
    }

    fn rnn_grad_data(&self,
                     src: &SharedTensor<T>,
                     src_gradient: &mut SharedTensor<T>,
                     output: &SharedTensor<T>,
                     output_gradient: &SharedTensor<T>,
                     rnn_config: &Self::RC,
                     weight: &SharedTensor<T>,
                     workspace: &mut SharedTensor<u8>)
                     -> Result<(), Error> {
        let src_dimensions = src.desc().clone();
        let sequence_descriptors = self.rnn_sequence_descriptors(
            src,
            *rnn_config.sequence_length(),
            rnn_config.hidden_size,
            src_dimensions[0] as i32,
        )?;
        let weight_desc = weight.cudnn_filter_desc()?;

        let src_mem = read!(src, self);
        let src_gradient_mem = read!(src_gradient, self);
        let weight_mem = read!(weight, self);
        let output_mem = read!(output, self);
        let workspace_mem = write_only!(workspace, self);
        unimplemented!()
        /*let reserve_space = rnn_config.training_reserve();
        dbg!("Running Backward Data");
        match CUDNN.rnn_backward_data::<f32>(
            rnn_config,
            sequence_descriptors.y_desc,
            trans!(output_mem),
            sequence_descriptors.dy_desc,
            //output_gradient,
            std::ptr::null_mut(),
            sequence_descriptors.dhy_desc,
            //final_hidden_gradient,
            std::ptr::null_mut(),
            sequence_descriptors.dcy_desc,
            //final_cell_gradient,
            std::ptr::null_mut(),
            &weight_desc,
            trans!(weight_mem),
            sequence_descriptors.hx_desc,
            std::ptr::null(),
            sequence_descriptors.cx_desc,
            std::ptr::null(),
            sequence_descriptors.dx_desc,
            trans_mut!(src_gradient_mem),
            sequence_descriptors.dhx_desc,
            std::ptr::null_mut(),
            sequence_descriptors.dcx_desc,
            std::ptr::null_mut(),
            trans_mut!(workspace_mem),
            *reserve_space.id_c(),
        ) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Operation("Unable to execute CUDA cuDNN RNN Backward Data"))),
        }*/
    }
}

impl<T> SigmoidPointwise<T> for Backend<Cuda>
    where T: Float + Default + DataTypeInfo
{
    fn sigmoid_pointwise(&self,
                         x: &mut SharedTensor<T>)
                         -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let x_desc = x.cudnn_tensor_desc_flat()?;
        let x_mem = read_write!(x, self);

        match CUDNN.sigmoid_forward(&CUDNN.init_activation().unwrap(),
        							&x_desc,
                                    trans!(x_mem),
                                    &x_desc,
                                    trans_mut!(x_mem),
                                    scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN Sigmoid Pointwise forward."))),
        }
    }

    fn sigmoid_pointwise_grad(&self,
                              x: &SharedTensor<T>,
                              x_diff: &mut SharedTensor<T>)
                              -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let x_desc = x.cudnn_tensor_desc_flat()?;
        let dx_desc = x_diff.cudnn_tensor_desc_flat()?;
        let x_mem = read!(x, self);
        let dx_mem = read_write!(x_diff, self);
        // TODO move config one level up
        match CUDNN.sigmoid_backward(&CUDNN.init_activation().unwrap(),
        							 &x_desc,
                                     trans!(x_mem),
                                     &dx_desc,
                                     trans!(dx_mem),
                                     &x_desc,
                                     trans!(x_mem),
                                     &dx_desc,
                                     trans_mut!(dx_mem),
                                     scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Operation("Unable to execute CUDA cuDNN Sigmoid Pointwise backward."))),
        }
    }
}

impl<T> Relu<T> for Backend<Cuda>
    where T: Float + Default + DataTypeInfo
{
    fn relu(&self,
            x: &SharedTensor<T>,
            result: &mut SharedTensor<T>)
            -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let r_desc = result.cudnn_tensor_desc_flat()?;
        let x_mem = read!(x, self);
        let r_mem = write_only!(result, self);
        match CUDNN.relu_forward(&CUDNN.init_activation().unwrap(),
        						&x.cudnn_tensor_desc_flat()?,
                                 trans!(x_mem),
                                 &r_desc,
                                 trans_mut!(r_mem),
                                 scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN Activation relu Forward."))),
        }
    }

    fn relu_grad(&self,
                 x: &SharedTensor<T>,
                 x_diff: &SharedTensor<T>,
                 result: &SharedTensor<T>,
                 result_diff: &mut SharedTensor<T>)
                 -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let dr_desc = result_diff.cudnn_tensor_desc_flat()?;
        let x_mem = read!(x, self);
        let dx_mem = read!(x_diff, self);
        let r_mem = read!(result, self);
        let dr_mem = write_only!(result_diff, self);

        match CUDNN.relu_backward(&CUDNN.init_activation().unwrap(),
        						&x.cudnn_tensor_desc_flat()?,
                                  trans!(x_mem),
                                  &x_diff.cudnn_tensor_desc_flat()?,
                                  trans!(dx_mem),
                                  &result.cudnn_tensor_desc_flat()?,
                                  trans!(r_mem),
                                  &dr_desc,
                                  trans_mut!(dr_mem),
                                  scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN Activation relu Backward."))),
        }
    }
}

impl<T> ReluPointwise<T> for Backend<Cuda>
    where T: Float + Default + DataTypeInfo
{
    fn relu_pointwise(&self,
                      x: &mut SharedTensor<T>)
                      -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let x_desc = x.cudnn_tensor_desc_flat()?;
        let x_mem = read_write!(x, self);

        match CUDNN.relu_forward(&CUDNN.init_activation().unwrap(),
        						&x_desc,
                                 trans!(x_mem),
                                 &x_desc,
                                 trans_mut!(x_mem),
                                 scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN ReLU Pointwise forward."))),
        }
    }

    fn relu_pointwise_grad(&self,
                           x: &SharedTensor<T>,
                           x_diff: &mut SharedTensor<T>)
                           -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let x_desc = x.cudnn_tensor_desc_flat()?;
        let dx_desc = x_diff.cudnn_tensor_desc_flat()?;
        let x_mem = read!(x, self);
        let dx_mem = read_write!(x_diff, self);

        match CUDNN.relu_backward(&CUDNN.init_activation().unwrap(),
        						&x_desc,
                                  trans!(x_mem),
                                  &dx_desc,
                                  trans!(dx_mem),
                                  &x_desc,
                                  trans!(x_mem),
                                  &dx_desc,
                                  trans_mut!(dx_mem),
                                  scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN ReLU Pointwise backward."))),
        }
    }
}

impl<T> Tanh<T> for Backend<Cuda>
    where T: Float + Default + DataTypeInfo
{
    fn tanh(&self,
            x: &SharedTensor<T>,
            result: &mut SharedTensor<T>)
            -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let r_desc = result.cudnn_tensor_desc_flat()?;
        let x_mem = read!(x, self);
        let r_mem = write_only!(result, self);
        match CUDNN.tanh_forward(&CUDNN.init_activation().unwrap(),
        						&x.cudnn_tensor_desc_flat()?,
                                 trans!(x_mem),
                                 &r_desc,
                                 trans_mut!(r_mem),
                                 scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN Activation tanh Forward."))),
        }
    }

    fn tanh_grad(&self,
                 x: &SharedTensor<T>,
                 x_diff: &SharedTensor<T>,
                 result: &SharedTensor<T>,
                 result_diff: &mut SharedTensor<T>)
                 -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let dr_desc = result_diff.cudnn_tensor_desc_flat()?;
        let x_mem = read!(x, self);
        let dx_mem = read!(x_diff, self);
        let r_mem = read!(result, self);
        let dr_mem = write_only!(result_diff, self);
        match CUDNN.tanh_backward(&CUDNN.init_activation().unwrap(),
        						&x.cudnn_tensor_desc_flat()?,
                                  trans!(x_mem),
                                  &x_diff.cudnn_tensor_desc_flat()?,
                                  trans!(dx_mem),
                                  &result.cudnn_tensor_desc_flat()?,
                                  trans!(r_mem),
                                  &dr_desc,
                                  trans_mut!(dr_mem),
                                  scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN Activation tanh Backward."))),
        }
    }
}

impl<T> TanhPointwise<T> for Backend<Cuda>
    where T: Float + Default + DataTypeInfo
{
    fn tanh_pointwise(&self,
                      x: &mut SharedTensor<T>)
                      -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let x_desc = x.cudnn_tensor_desc_flat()?;
        let x_mem = read_write!(x, self);
        match CUDNN.tanh_forward(&CUDNN.init_activation().unwrap(),
        						&x_desc,
                                 trans!(x_mem),
                                 &x_desc,
                                 trans_mut!(x_mem),
                                 scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN Tanh Pointwise forward."))),
        }
    }

    fn tanh_pointwise_grad(&self,
                           x: &SharedTensor<T>,
                           x_diff: &mut SharedTensor<T>)
                           -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let x_desc = x.cudnn_tensor_desc_flat()?;
        let dx_desc = x_diff.cudnn_tensor_desc_flat()?;
        let x_mem = read!(x, self);
        let dx_mem = read_write!(x_diff, self);
        match CUDNN.tanh_backward(&CUDNN.init_activation().unwrap(),
        						&x_desc,
                                  trans!(x_mem),
                                  &dx_desc,
                                  trans!(dx_mem),
                                  &x_desc,
                                  trans!(x_mem),
                                  &dx_desc,
                                  trans_mut!(dx_mem),
                                  scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN Tanh Pointwise backward."))),
        }
    }
}

impl<T> Softmax<T> for Backend<Cuda>
    where T: Float + Default + DataTypeInfo
{
    fn softmax(&self,
               x: &SharedTensor<T>,
               result: &mut SharedTensor<T>)
               -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let r_desc = result.cudnn_tensor_desc_softmax()?;
        let x_mem = read!(x, self);
        let r_mem = write_only!(result, self);
        match CUDNN.softmax_forward(&x.cudnn_tensor_desc_softmax()?,
                                    trans!(x_mem),
                                    &r_desc,
                                    trans_mut!(r_mem),
                                    scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN softmax Forward."))),
        }
    }

    fn softmax_grad(&self,
                    x: &SharedTensor<T>,
                    x_diff: &SharedTensor<T>,
                    result_diff: &mut SharedTensor<T>)
                    -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let dr_desc = result_diff.cudnn_tensor_desc_softmax()?;
        let x_mem = read!(x, self);
        let dx_mem = read!(x_diff, self);
        let dr_mem = write_only!(result_diff, self);
        match CUDNN.softmax_backward(&x.cudnn_tensor_desc_softmax()?,
                                     trans!(x_mem),
                                     &x_diff.cudnn_tensor_desc_softmax()?,
                                     trans!(dx_mem),
                                     &dr_desc,
                                     trans_mut!(dr_mem),
                                     scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN softmax Backward."))),
        }
    }
}

impl<T> LogSoftmax<T> for Backend<Cuda>
    where T: Float + Default + DataTypeInfo
{
    fn log_softmax(&self,
                   x: &SharedTensor<T>,
                   result: &mut SharedTensor<T>)
                   -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let r_desc = result.cudnn_tensor_desc_softmax()?;
        let x_mem = read!(x, self);
        let r_mem = write_only!(result, self);
        match CUDNN.log_softmax_forward(&x.cudnn_tensor_desc_softmax()?,
                                    trans!(x_mem),
                                    &r_desc,
                                    trans_mut!(r_mem),
                                    scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN softmax Forward."))),
        }
    }
    fn log_softmax_grad(&self,
                    x: &SharedTensor<T>,
                    x_diff: &SharedTensor<T>,
                    result_diff: &mut SharedTensor<T>)
                    -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let dr_desc = result_diff.cudnn_tensor_desc_softmax()?;
        let x_mem = read!(x, self);
        let dx_mem = read!(x_diff, self);
        let dr_mem = write_only!(result_diff, self);
        match CUDNN.log_softmax_backward(&x.cudnn_tensor_desc_softmax()?,
                                     trans!(x_mem),
                                     &x_diff.cudnn_tensor_desc_softmax()?,
                                     trans!(dx_mem),
                                     &dr_desc,
                                     trans_mut!(dr_mem),
                                     scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN log softmax Backward."))),
        }
    }

}

impl<T> LRN<T> for Backend<Cuda>
    where T: Float + Default + DataTypeInfo
{
    fn new_lrn_config(&self,
                      n: u32,
                      alpha: f64,
                      beta: f64,
                      k: f64)
                      -> Result<Self::CLRN, Error> {
        Ok(CUDNN.init_normalization(n, alpha, beta, k).unwrap())
    }

    fn lrn(&self,
           x: &SharedTensor<T>,
           result: &mut SharedTensor<T>,
           config: &Self::CLRN)
           -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let r_desc = result.cudnn_tensor_desc()?;
        let x_mem = read!(x, self);
        let r_mem = write_only!(result, self);
        match CUDNN.lrn_forward(config,
                                &x.cudnn_tensor_desc()?,
                                trans!(x_mem),
                                &r_desc,
                                trans_mut!(r_mem),
                                scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN Activation lrn Forward."))),
        }
    }

    #[allow(unused_variables)]
    fn lrn_grad(&self,
                x: &SharedTensor<T>,
                x_diff: &SharedTensor<T>,
                result: &SharedTensor<T>,
                result_diff: &mut SharedTensor<T>,
                config: &Self::CLRN)
                -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let dr_desc = result_diff.cudnn_tensor_desc()?;
        let x_mem = read!(x, self);
        let dx_mem = read!(x_diff, self);
        let r_mem = read!(result, self);
        let dr_mem = write_only!(result_diff, self);
        match CUDNN.lrn_backward(config,
                                 &x.cudnn_tensor_desc()?,
                                 trans!(x_mem),
                                 &x_diff.cudnn_tensor_desc()?,
                                 trans!(dx_mem),
                                 &result.cudnn_tensor_desc()?,
                                 trans!(r_mem),
                                 &dr_desc,
                                 trans_mut!(dr_mem),
                                 scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN Activation lrn Backward."))),
        }
    }
}

impl<T> Pooling<T> for Backend<Cuda>
    where T: Float + Default + DataTypeInfo
{
    fn new_pooling_config(&self,
                          window: &[i32],
                          stride: &[i32],
                          padding: &[i32])
                          -> Result<Self::CPOOL, Error> {
        let pooling_avg = crate::cudnn::PoolingDescriptor::new(crate::cudnn::cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, window, padding, stride).unwrap();
        let pooling_max =
            crate::cudnn::PoolingDescriptor::new(crate::cudnn::cudnnPoolingMode_t::CUDNN_POOLING_MAX,
                                            window,
                                            padding,
                                            stride)
                    .unwrap();
        Ok(crate::cudnn::utils::PoolingConfig::new(pooling_avg, pooling_max))
    }

    fn pooling_max(&self,
                   x: &SharedTensor<T>,
                   result: &mut SharedTensor<T>,
                   config: &Self::CPOOL)
                   -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();

        let r_desc = result.cudnn_tensor_desc()?;
        let x_mem = read!(x, self);
        let r_mem = write_only!(result, self);
        match CUDNN.pooling_max_forward(config,
                                        &x.cudnn_tensor_desc()?,
                                        trans!(x_mem),
                                        &r_desc,
                                        trans_mut!(r_mem),
                                        scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN max pooling Forward."))),
        }
    }

    #[allow(unused_variables)]
    fn pooling_max_grad(&self,
                        x: &SharedTensor<T>,
                        x_diff: &SharedTensor<T>,
                        result: &SharedTensor<T>,
                        result_diff: &mut SharedTensor<T>,
                        config: &Self::CPOOL)
                        -> Result<(), Error> {

        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let dr_desc = result_diff.cudnn_tensor_desc()?;
        let x_mem = read!(x, self);
        let dx_mem = read!(x_diff, self);
        let r_mem = read!(result, self);
        let dr_mem = write_only!(result_diff, self);
        match CUDNN.pooling_max_backward(config,
                                         &x.cudnn_tensor_desc()?,
                                         trans!(x_mem),
                                         &x_diff.cudnn_tensor_desc()?,
                                         trans!(dx_mem),
                                         &result.cudnn_tensor_desc()?,
                                         trans!(r_mem),
                                         &dr_desc,
                                         trans_mut!(dr_mem),
                                         scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN max pooling Backward."))),
        }
    }

    fn pooling_avg(&self,
                   x: &SharedTensor<T>,
                   result: &mut SharedTensor<T>,
                   config: &Self::CPOOL)
                   -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let r_desc = result.cudnn_tensor_desc()?;
        let x_mem = read!(x, self);
        let r_mem = write_only!(result, self);
        match CUDNN.pooling_avg_forward(config,
                                        &x.cudnn_tensor_desc()?,
                                        trans!(x_mem),
                                        &r_desc,
                                        trans_mut!(r_mem),
                                        scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN avg pooling Forward."))),
        }
    }

    #[allow(unused_variables)]
    fn pooling_avg_grad(&self,
                        x: &SharedTensor<T>,
                        x_diff: &SharedTensor<T>,
                        result: &SharedTensor<T>,
                        result_diff: &mut SharedTensor<T>,
                        config: &Self::CPOOL)
                        -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> = crate::cudnn::utils::ScalParams::default();
        let dr_desc = result_diff.cudnn_tensor_desc()?;
        let x_mem = read!(x, self);
        let dx_mem = read!(x_diff, self);
        let r_mem = read!(result, self);
        let dr_mem = write_only!(result_diff, self);
        match CUDNN.pooling_avg_backward(config,
                                         &x.cudnn_tensor_desc()?,
                                         trans!(x_mem),
                                         &x_diff.cudnn_tensor_desc()?,
                                         trans!(dx_mem),
                                         &result.cudnn_tensor_desc()?,
                                         trans!(r_mem),
                                         &dr_desc,
                                         trans_mut!(dr_mem),
                                         scal_params) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN avg pooling Backward."))),
        }
    }
}



impl<T> Dropout<T> for Backend<Cuda>
    where T: Float + Default + DataTypeInfo
{
    fn new_dropout_config(&self,
                      probability: f32,
                      seed: u64,
                      )
                      -> Result<Self::CDROP, Error> {
        Ok(CUDNN.init_dropout(probability, seed).unwrap())
    }

    fn dropout(&self,
           x: &SharedTensor<T>,
           result: &mut SharedTensor<T>,
           config: &Self::CDROP)
           -> Result<(), Error> {
        let r_desc = result.cudnn_tensor_desc()?;
        let x_mem = read!(x, self);
        let r_mem = write_only!(result, self);
        match CUDNN.dropout_forward::<f32>(config,
                                &x.cudnn_tensor_desc()?,
                                trans!(x_mem),
                                &r_desc,
                                trans_mut!(r_mem),
                                ) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN Dropout Forward."))),
        }
    }

    #[allow(unused_variables)]
    fn dropout_grad(&self,
                x: &SharedTensor<T>,
                x_diff: &SharedTensor<T>,
                result: &SharedTensor<T>,
                result_diff: &mut SharedTensor<T>,
                config: &Self::CDROP)
                -> Result<(), Error> {
        // TODO what to do with the gradient? should be all zeroes since this is supposed to be a `nop` but I am not 100% sure about the nv implementations
        // let dr_desc = result_diff.cudnn_tensor_desc()?;
        // let x_mem = read!(x, self);
        // let dx_mem = read!(x_diff, self);
        // let r_mem = write_only!(result, self);
        // let dr_mem = write_only!(result_diff, self);
        // match CUDNN.dropout_backward::<f32>(config,
        //                          &x.cudnn_tensor_desc()?,
        //                          trans!(x_mem),
        //                          &result.cudnn_tensor_desc()?,
        //                          trans_mut!(r_mem)) {
            // Ok(_) => Ok(()),
        //     Err(_) => Err(Error::Plugin(PluginError::Plugin("Unable to execute CUDA cuDNN Dropout Backward."))),
        // }
        Ok(())
    }
}