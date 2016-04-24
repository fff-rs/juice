//! Provides NN for a CUDA backend.
#![allow(missing_docs)]
use ::plugin::*;
use co::Error as CoError;
use co::prelude::*;
use co::plugin::Error as PluginError;
use cudnn::*;
use cudnn::utils::ScalParams;

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

    fn cudnn_convolution_desc(&self, filter: &SharedTensor<T>) -> Result<ConvolutionDescriptor, PluginError>;
}

macro_rules! impl_icudnndesc_for_sharedtensor {
    ($t:ty, $cutype:path) => (
        impl ICudnnDesc<$t> for SharedTensor<$t> {
            fn cudnn_tensor_desc(&self) -> Result<TensorDescriptor, PluginError> {
                TensorDescriptor::new(&self.desc().dims_i32().clone(),
                                      &self.desc().default_stride_i32().clone(),
                                      $cutype)
                    .map_err(|_| PluginError::Plugin(
                        "Unable to create CuDNN TensorDescriptor."))
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
                    _ => actual_desc
                };
                TensorDescriptor::new(&override_desc.dims_i32().clone(),
                                      &override_desc.default_stride_i32().clone(),
                                      $cutype)
                    .map_err(|_| PluginError::Plugin(
                        "Unable to create CuDNN TensorDescriptor."))
            }

            fn cudnn_tensor_desc_flat(&self) -> Result<TensorDescriptor, PluginError> {
                let actual_desc = self.desc().clone();
                let mut override_desc = match actual_desc.len() {
                    1 => vec![1, 1],
                    2 => vec![1],
                    _ => vec![]
                };
                for dim in actual_desc {
                    override_desc.push(dim);
                }
                TensorDescriptor::new(&override_desc.dims_i32().clone(),
                                      &override_desc.default_stride_i32().clone(),
                                      $cutype)
                    .map_err(|_| PluginError::Plugin(
                        "Unable to create CuDNN TensorDescriptor."))
            }

            fn cudnn_filter_desc(&self) -> Result<FilterDescriptor, PluginError> {
                FilterDescriptor::new(&self.desc().dims_i32().clone(), $cutype)
                    .map_err(|_| PluginError::Plugin(
                        "Unable to create CuDNN FilterDescriptor."))
            }

            fn cudnn_convolution_desc(&self, filter: &SharedTensor<$t>)
                                      -> Result<ConvolutionDescriptor, PluginError> {
                ConvolutionDescriptor::new(&self.desc().dims_i32().clone(),
                                           &filter.desc().default_stride_i32().clone(),
                                           $cutype)
                    .map_err(|_| PluginError::Plugin(
                        "Unable to create CuDNN ConvolutionDescriptor."))
            }
        }
    )
}

impl_icudnndesc_for_sharedtensor!(f32, ::cudnn::utils::DataType::Float);
impl_icudnndesc_for_sharedtensor!(f64, ::cudnn::utils::DataType::Double);

impl_oconf_for_cc!(f32, f64);
impl_oconf_for_clrn!(f32, f64);
impl_oconf_for_pooling!(f32, f64);

impl ConvForwardAlgo {
    /// Tries to return the matching cuDNN type for the enum value.
    fn as_cudnn(&self) -> Result<cudnnConvolutionFwdAlgo_t, CoError> {
        use ConvForwardAlgo::*;
        use ::cudnn::cudnnConvolutionFwdAlgo_t::*;
        Ok(match *self {
            Auto => return Err(CoError::Plugin(PluginError::Plugin(
                "Can't create cuDNN convolution forward algorithm from \
                 ConvForwardAlgo::Auto. Use `find_cudnn_algo` to find an algorithm."))),
            GEMM => CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
            ImplicitGEMM => CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            ImplicitPrecompiledGEMM => CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
            FFT => CUDNN_CONVOLUTION_FWD_ALGO_FFT,
            FFTTiling => CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
            Direct => CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        })
    }

    /// Returns the matching enum value for a cuDNN algo.
    fn from_cudnn(algo: &cudnnConvolutionFwdAlgo_t) -> ConvForwardAlgo {
        use ConvForwardAlgo::*;
        use ::cudnn::cudnnConvolutionFwdAlgo_t::*;
        match *algo {
            CUDNN_CONVOLUTION_FWD_ALGO_GEMM => GEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM => ImplicitGEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM => ImplicitPrecompiledGEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_FFT => FFT,
            CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING => FFTTiling,
            CUDNN_CONVOLUTION_FWD_ALGO_DIRECT => Direct,
        }
    }

    /// Try to find best algorithm for a operation that uses the provided descriptors.
    fn find_cudnn_algo(
        &self,
        filter_desc: &FilterDescriptor,
        conv_desc: &ConvolutionDescriptor,
        src_desc: &TensorDescriptor,
        dest_desc: &TensorDescriptor,
    ) -> Result<ConvForwardAlgo, CoError> {
        if !self.is_auto() {
            return Ok(*self);
        }
        let algos = API::find_convolution_forward_algorithm(
            *CUDNN.id_c(), *filter_desc.id_c(), *conv_desc.id_c(),
            *src_desc.id_c(), *dest_desc.id_c()).unwrap();
        let algo = match algos.len() {
            0 => return Err(CoError::Plugin(PluginError::Operation(
                "Unable to find CUDA cuDNN convolution forward algorithm."))),
            _ => algos[0].algo
        };
        Ok(ConvForwardAlgo::from_cudnn(&algo))
    }
}

impl ConvBackwardFilterAlgo {
    /// Tries to return the matching cuDNN type for the enum value.
    fn as_cudnn(&self) -> Result<cudnnConvolutionBwdFilterAlgo_t, CoError> {
        use ConvBackwardFilterAlgo::*;
        use ::cudnn::cudnnConvolutionBwdFilterAlgo_t::*;
        Ok(match *self {
            Auto => return Err(CoError::Plugin(PluginError::Plugin(
                "Can't create cuDNN convolution backward filter algorithm from \
                 ConvBackwardFilterAlgo::Auto. Use `find_cudnn_algo` to find an \
                 algorithm."))),
            ImplicitGEMM => CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
            ImplicitGEMMSum => CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
            ImplicitPrecompiledGEMMSum => CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
            FFT => CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
        })
    }

    /// Returns the matching enum value for a cuDNN algo.
    fn from_cudnn(algo: &cudnnConvolutionBwdFilterAlgo_t) -> ConvBackwardFilterAlgo {
        use ConvBackwardFilterAlgo::*;
        use ::cudnn::cudnnConvolutionBwdFilterAlgo_t::*;
        match *algo {
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 => ImplicitGEMMSum,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 => ImplicitGEMM,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT => FFT,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 => ImplicitPrecompiledGEMMSum,
        }
    }

    /// Try to find best algorithm for a operation that uses the provided descriptors.
    fn find_cudnn_algo(
        &self,
        filter_desc: &FilterDescriptor,
        conv_desc: &ConvolutionDescriptor,
        src_desc: &TensorDescriptor,
        dest_desc: &TensorDescriptor,
    ) -> Result<ConvBackwardFilterAlgo, CoError> {
        if !self.is_auto() {
            return Ok(*self);
        }
        let algos = API::find_convolution_backward_filter_algorithm(
            *CUDNN.id_c(), *filter_desc.id_c(), *conv_desc.id_c(),
            *src_desc.id_c(), *dest_desc.id_c()).unwrap();
        let algo = match algos.len() {
            0 => return Err(CoError::Plugin(PluginError::Operation(
                "Unable to find CUDA cuDNN convolution backward filter algorithm."))),
            _ => algos[0].algo
        };
        Ok(ConvBackwardFilterAlgo::from_cudnn(&algo))
    }
}

impl ConvBackwardDataAlgo {
    /// Tries to return the matching cuDNN type for the enum value.
    fn as_cudnn(&self) -> Result<cudnnConvolutionBwdDataAlgo_t, CoError> {
        use ConvBackwardDataAlgo::*;
        use ::cudnn::cudnnConvolutionBwdDataAlgo_t::*;
        Ok(match *self {
            Auto => return Err(CoError::Plugin(PluginError::Plugin(
                "Can't create cuDNN convolution backward data algorithm from \
                 ConvBackwardDataAlgo::Auto. Use `find_cudnn_algo` to find \
                 an algorithm."))),
            ImplicitGEMM => CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
            ImplicitGEMMSum => CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
            FFT => CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
            FFTTiling => CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
        })
    }

    /// Returns the matching enum value for a cuDNN algo.
    fn from_cudnn(algo: &cudnnConvolutionBwdDataAlgo_t) -> ConvBackwardDataAlgo {
        use ConvBackwardDataAlgo::*;
        use ::cudnn::cudnnConvolutionBwdDataAlgo_t::*;
        match *algo {
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 => ImplicitGEMMSum,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 => ImplicitGEMM,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT => FFT,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING => FFTTiling,
        }
    }

    /// Try to find best algorithm for a operation that uses the provided descriptors.
    fn find_cudnn_algo(
        &self,
        filter_desc: &FilterDescriptor,
        conv_desc: &ConvolutionDescriptor,
        src_desc: &TensorDescriptor,
        dest_desc: &TensorDescriptor,
    ) -> Result<ConvBackwardDataAlgo, CoError> {
        if !self.is_auto() {
            return Ok(*self);
        }
        let algos = API::find_convolution_backward_data_algorithm(
            *CUDNN.id_c(), *filter_desc.id_c(), *conv_desc.id_c(),
            *src_desc.id_c(), *dest_desc.id_c()).unwrap();

        let algo = match algos.len() {
            0 => return Err(CoError::Plugin(PluginError::Operation(
                "Unable to find CUDA cuDNN convolution backward data algorithm."))),
            _ => algos[0].algo
        };
        Ok(ConvBackwardDataAlgo::from_cudnn(&algo))
    }
}

macro_rules! impl_convolution_for_cuda_backend {
    ($t:ty, $cutype:path) => (
        impl ConvolutionConfig<$t> for ::cudnn::utils::ConvolutionConfig {
            fn workspace_size(&self) -> usize {
                *self.largest_workspace_size()
            }
        }

        impl Convolution<$t> for Backend<Cuda> {
            fn new_convolution_config(
                &self,
                src: &SharedTensor<$t>,
                dest: &SharedTensor<$t>,
                filter: &SharedTensor<$t>,
                algo_fwd: ConvForwardAlgo,
                algo_bwd_filter: ConvBackwardFilterAlgo,
                algo_bwd_data: ConvBackwardDataAlgo,
                stride: &[i32],
                zero_padding: &[i32],
            ) -> Result<Self::CC, CoError> {
                let src_desc = try!(src.cudnn_tensor_desc());
                let dest_desc = try!(dest.cudnn_tensor_desc());
                let filter_desc = try!(filter.cudnn_filter_desc());
                let conv_desc = ConvolutionDescriptor::new(
                    zero_padding, stride, $cutype).unwrap();

                let useable_algo_fwd = try!(algo_fwd.find_cudnn_algo(
                    &filter_desc, &conv_desc, &src_desc, &dest_desc));
                let useable_algo_bwd_filter = try!(algo_bwd_filter.find_cudnn_algo(
                    &filter_desc, &conv_desc, &src_desc, &dest_desc));
                let useable_algo_bwd_data = try!(algo_bwd_data.find_cudnn_algo(
                    &filter_desc, &conv_desc, &src_desc, &dest_desc));

                let mut workspace_size_fwd =
                    API::get_convolution_forward_workspace_size(
                        *CUDNN.id_c(), useable_algo_fwd.as_cudnn().unwrap(),
                        *filter_desc.id_c(), *conv_desc.id_c(), *src_desc.id_c(),
                        *dest_desc.id_c()).unwrap();
                let mut workspace_size_bwd_filter =
                    API::get_convolution_backward_filter_workspace_size(
                        *CUDNN.id_c(), useable_algo_bwd_filter.as_cudnn().unwrap(),
                        *filter_desc.id_c(), *conv_desc.id_c(), *src_desc.id_c(),
                        *dest_desc.id_c()).unwrap();
                let mut workspace_size_bwd_data =
                    API::get_convolution_backward_data_workspace_size(
                        *CUDNN.id_c(), useable_algo_bwd_data.as_cudnn().unwrap(),
                        *filter_desc.id_c(), *conv_desc.id_c(), *src_desc.id_c(),
                        *dest_desc.id_c()).unwrap();

                if workspace_size_fwd == 0 {
                    workspace_size_fwd = 8;
                }
                if workspace_size_bwd_filter == 0 {
                    workspace_size_bwd_filter = 8;
                }
                if workspace_size_bwd_data == 0 {
                    workspace_size_bwd_data = 8;
                }

                Ok(::cudnn::utils::ConvolutionConfig::new(
                    useable_algo_fwd.as_cudnn().unwrap(), workspace_size_fwd,
                    useable_algo_bwd_filter.as_cudnn().unwrap(), workspace_size_bwd_filter,
                    useable_algo_bwd_data.as_cudnn().unwrap(), workspace_size_bwd_data,
                    conv_desc, filter_desc)
                )
            }

            impl_ops_convolution_for!($t, Backend<Cuda>);
        }
    )
}

impl NN<f32> for Backend<Cuda> {
    type CC = utils::ConvolutionConfig;
    type CLRN = utils::NormalizationConfig;
    type CPOOL = utils::PoolingConfig;

    fn init_nn() { let _ = CUDNN.id_c(); }
    fn device(&self) -> &DeviceType { self.device() }
}

impl_convolution_for_cuda_backend!(f32, ::cudnn::utils::DataType::Float);
impl_ops_sigmoid_for!(f32, Backend<Cuda>);
impl_ops_relu_for!(f32, Backend<Cuda>);
impl_ops_tanh_for!(f32, Backend<Cuda>);
impl_ops_softmax_for!(f32, Backend<Cuda>);
impl_ops_log_softmax_for!(f32, Backend<Cuda>);
impl_ops_lrn_for!(f32, Backend<Cuda>);
impl_ops_pooling_for!(f32, Backend<Cuda>);

impl NN<f64> for Backend<Cuda> {
    type CC = utils::ConvolutionConfig;
    type CLRN = utils::NormalizationConfig;
    type CPOOL = utils::PoolingConfig;

    fn init_nn() { let _ = CUDNN.id_c(); }
    fn device(&self) -> &DeviceType { self.device() }
}

impl_convolution_for_cuda_backend!(f64, ::cudnn::utils::DataType::Double);
impl_ops_sigmoid_for!(f64, Backend<Cuda>);
impl_ops_relu_for!(f64, Backend<Cuda>);
impl_ops_tanh_for!(f64, Backend<Cuda>);
impl_ops_softmax_for!(f64, Backend<Cuda>);
impl_ops_log_softmax_for!(f64, Backend<Cuda>);
impl_ops_lrn_for!(f64, Backend<Cuda>);
impl_ops_pooling_for!(f64, Backend<Cuda>);
