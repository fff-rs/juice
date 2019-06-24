//! Provides NN for a CUDA backend.
#![allow(missing_docs)]

use crate::co::Error;
use crate::co::plugin::Error as PluginError;
use crate::co::plugin::numeric_helpers::Float;
use crate::co::prelude::*;
use crate::cudnn::*;

pub use crate::cudnn::utils::DataTypeInfo;
use crate::plugin::*;

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
}

impl<T> NN<T> for Backend<Cuda>
    where T: Float + DataTypeInfo
{
    type CC = utils::ConvolutionConfig;
    type CLRN = utils::NormalizationConfig;
    type CPOOL = utils::PoolingConfig;
    type CDROP = utils::DropoutConfig;

    fn init_nn() {
        let _ = CUDNN.id_c();
    }
}

impl<'a, T> NNOperationConfig<T> for utils::ConvolutionConfig where T: Float + DataTypeInfo {}
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
        *self.largest_workspace_size()
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
