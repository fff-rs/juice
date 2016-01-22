//! Provides NN for a CUDA backend.
#![allow(missing_docs)]
use ::plugin::*;
use co::backend::Backend;
use co::device::DeviceType;
use co::tensor::{SharedTensor, ITensorDesc};
use co::plugin::Error as PluginError;
use co::frameworks::cuda::Cuda;
use cudnn::*;

#[macro_use]
pub mod helper;

lazy_static! {
    static ref CUDNN: Cudnn = Cudnn::new().unwrap();
}

pub trait ICudnnDesc<T> {
    fn cudnn_tensor_desc(&self) -> Result<TensorDescriptor, PluginError>;

    fn cudnn_filter_desc(&self) -> Result<FilterDescriptor, PluginError>;

    fn cudnn_convolution_desc(&self, filter: &SharedTensor<T>) -> Result<ConvolutionDescriptor, PluginError>;
}

impl ICudnnDesc<f32> for SharedTensor<f32> {
    fn cudnn_tensor_desc(&self) -> Result<TensorDescriptor, PluginError> {
        match TensorDescriptor::new(&self.desc().dims_i32().clone(), &self.desc().default_stride_i32().clone(), utils::DataType::Float) {
            Ok(desc) => Ok(desc),
            Err(err) => {
                println!("{:?}", err);
                Err(PluginError::Plugin("Unable to create CuDNN TensorDescriptor."))
            }
        }
    }

    fn cudnn_filter_desc(&self) -> Result<FilterDescriptor, PluginError> {
        match FilterDescriptor::new(&self.desc().dims_i32().clone(), utils::DataType::Float) {
            Ok(desc) => Ok(desc),
            Err(err) => {
                println!("{:?}", err);
                Err(PluginError::Plugin("Unable to create CuDNN FilterDescriptor."))
            }
        }
    }

    fn cudnn_convolution_desc(&self, filter: &SharedTensor<f32>) -> Result<ConvolutionDescriptor, PluginError> {
        match ConvolutionDescriptor::new(&self.desc().dims_i32().clone(), &filter.desc().default_stride_i32().clone(), utils::DataType::Float) {
            Ok(desc) => Ok(desc),
            Err(err) => {
                println!("{:?}", err);
                Err(PluginError::Plugin("Unable to create CuDNN ConvolutionDescriptor."))
            }
        }
    }
}

impl ICudnnDesc<f64> for SharedTensor<f64> {
    fn cudnn_tensor_desc(&self) -> Result<TensorDescriptor, PluginError> {
        match TensorDescriptor::new(&self.desc().dims_i32().clone(), &self.desc().default_stride_i32().clone(), utils::DataType::Double) {
            Ok(desc) => Ok(desc),
            Err(err) => {
                println!("{:?}", err);
                Err(PluginError::Plugin("Unable to create CuDNN TensorDescriptor."))
            }
        }
    }

    fn cudnn_filter_desc(&self) -> Result<FilterDescriptor, PluginError> {
        match FilterDescriptor::new(&self.desc().dims_i32().clone(), utils::DataType::Double) {
            Ok(desc) => Ok(desc),
            Err(err) => {
                println!("{:?}", err);
                Err(PluginError::Plugin("Unable to create CuDNN FilterDescriptor."))
            }
        }
    }

    fn cudnn_convolution_desc(&self, filter: &SharedTensor<f64>) -> Result<ConvolutionDescriptor, PluginError> {
        match ConvolutionDescriptor::new(&self.desc().dims_i32().clone(), &filter.desc().default_stride_i32().clone(), utils::DataType::Double) {
            Ok(desc) => Ok(desc),
            Err(err) => {
                println!("{:?}", err);
                Err(PluginError::Plugin("Unable to create CuDNN ConvolutionDescriptor."))
            }
        }
    }
}

impl_oconf_for_cc!(f32, f64);
impl_oconf_for_clrn!(f32, f64);
impl_oconf_for_pooling!(f32, f64);

impl NN<f32> for Backend<Cuda> {
    type CC = utils::ConvolutionConfig;
    type CLRN = utils::NormalizationConfig;
    type CPOOL = utils::PoolingConfig;

    fn init_nn() { let _ = CUDNN.id_c(); }
    fn device(&self) -> &DeviceType { self.device() }
}

impl Convolution<f32> for Backend<Cuda> {
    fn new_convolution_config(
        &self,
        src: &::co::tensor::SharedTensor<f32>,
        dest: &::co::tensor::SharedTensor<f32>,
        filter: &mut ::co::tensor::SharedTensor<f32>,
        stride: &[i32],
        zero_padding: &[i32],
    ) -> Result<Self::CC, ::co::error::Error> {
        let src_desc = try!(src.cudnn_tensor_desc());
        let dest_desc = try!(dest.cudnn_tensor_desc());
        let filter_desc = try!(filter.cudnn_filter_desc());
        let conv_desc = ::cudnn::ConvolutionDescriptor::new(zero_padding, stride, ::cudnn::utils::DataType::Float).unwrap();

        // let algos_fwd = API::find_convolution_forward_algorithm(*CUDNN.id_c(), *filter_desc.id_c(), *src_desc.id_c(), *conv_desc.id_c(), *dest_desc.id_c()).unwrap();
        // let algo_fwd = match algos_fwd.len() {
        //     0 => return Err(::co::error::Error::Plugin(::co::plugin::Error::Operation("Unable to load CUDA cuDNN convolution forward algorithm."))),
        //     _ => algos_fwd[0].algo
        // };
        let algo_fwd = ::cudnn::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        let (workspace_fwd, workspace_size_fwd) = match algo_fwd {
            ::cudnn::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM => (::co::frameworks::cuda::Memory::from_c(0), 0),
            _ => {
                let workspace_size_fwd = API::get_convolution_forward_workspace_size(*CUDNN.id_c(), algo_fwd, *filter_desc.id_c(), *src_desc.id_c(), *conv_desc.id_c(), *dest_desc.id_c()).unwrap();
                let workspace_forward = ::co::frameworks::cuda::Memory::new(workspace_size_fwd).unwrap();
                (workspace_forward, workspace_size_fwd)
            }
        };

        // let algos_bwd = API::find_convolution_backward_data_algorithm(*CUDNN.id_c(), *filter_desc.id_c(), *src_desc.id_c(), *conv_desc.id_c(), *dest_desc.id_c()).unwrap();
        // let algo_bwd = match algos_bwd.len() {
        //     0 => return Err(::co::error::Error::Plugin(::co::plugin::Error::Operation("Unable to load CUDA cuDNN convolution backward algorithm."))),
        //     _ => algos_bwd[0].algo
        // };
        let algo_bwd = ::cudnn::cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
        let (workspace_bwd, workspace_size_bwd) = match algo_bwd {
            ::cudnn::cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 => (::co::frameworks::cuda::Memory::from_c(0), 0),
            ::cudnn::cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 => (::co::frameworks::cuda::Memory::from_c(0), 0),
            _ => {
                    let workspace_size_bwd = API::get_convolution_backward_data_workspace_size(*CUDNN.id_c(), algo_bwd, *filter_desc.id_c(), *src_desc.id_c(), *conv_desc.id_c(), *dest_desc.id_c()).unwrap();
                    let workspace_backward = ::co::frameworks::cuda::Memory::new(workspace_size_bwd).unwrap();
                    (workspace_backward, workspace_size_bwd)
            }
        };

        Ok(
            ::cudnn::utils::ConvolutionConfig::new(
                algo_fwd, workspace_fwd, workspace_size_fwd,
                algo_bwd, workspace_bwd, workspace_size_bwd,
                conv_desc, filter_desc, filter.remove_copy(self.device()).unwrap().into_cuda().unwrap()
            )
        )
    }

    impl_ops_convolution_for!(f32, Backend<Cuda>);
}

impl_ops_sigmoid_for!(f32, Backend<Cuda>);
impl_ops_relu_for!(f32, Backend<Cuda>);
impl_ops_tanh_for!(f32, Backend<Cuda>);
impl_ops_softmax_for!(f32, Backend<Cuda>);
impl_ops_lrn_for!(f32, Backend<Cuda>);
impl_ops_pooling_for!(f32, Backend<Cuda>);


impl NN<f64> for Backend<Cuda> {
    type CC = utils::ConvolutionConfig;
    type CLRN = utils::NormalizationConfig;
    type CPOOL = utils::PoolingConfig;

    fn init_nn() { let _ = CUDNN.id_c(); }
    fn device(&self) -> &DeviceType { self.device() }
}

impl Convolution<f64> for Backend<Cuda> {
    fn new_convolution_config(
        &self,
        src: &::co::tensor::SharedTensor<f64>,
        dest: &::co::tensor::SharedTensor<f64>,
        filter: &mut ::co::tensor::SharedTensor<f64>,
        stride: &[i32],
        zero_padding: &[i32],
    ) -> Result<Self::CC, ::co::error::Error> {
        let src_desc = try!(src.cudnn_tensor_desc());
        let dest_desc = try!(dest.cudnn_tensor_desc());
        let filter_desc = try!(filter.cudnn_filter_desc());
        let conv_desc = ::cudnn::ConvolutionDescriptor::new(zero_padding, stride, ::cudnn::utils::DataType::Double).unwrap();

        // let algos_fwd = API::find_convolution_forward_algorithm(*CUDNN.id_c(), *filter_desc.id_c(), *src_desc.id_c(), *conv_desc.id_c(), *dest_desc.id_c()).unwrap();
        // let algo_fwd = match algos_fwd.len() {
        //     0 => return Err(::co::error::Error::Plugin(::co::plugin::Error::Operation("Unable to load CUDA cuDNN convolution forward algorithm."))),
        //     _ => algos_fwd[0].algo
        // };
        let algo_fwd = ::cudnn::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        let (workspace_fwd, workspace_size_fwd) = match algo_fwd {
            ::cudnn::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM => (::co::frameworks::cuda::Memory::from_c(0), 0),
            _ => {
                let workspace_size_fwd = API::get_convolution_forward_workspace_size(*CUDNN.id_c(), algo_fwd, *filter_desc.id_c(), *src_desc.id_c(), *conv_desc.id_c(), *dest_desc.id_c()).unwrap();
                let workspace_forward = ::co::frameworks::cuda::Memory::new(workspace_size_fwd).unwrap();
                (workspace_forward, workspace_size_fwd)
            }
        };

        // let algos_bwd = API::find_convolution_backward_data_algorithm(*CUDNN.id_c(), *filter_desc.id_c(), *src_desc.id_c(), *conv_desc.id_c(), *dest_desc.id_c()).unwrap();
        // let algo_bwd = match algos_bwd.len() {
        //     0 => return Err(::co::error::Error::Plugin(::co::plugin::Error::Operation("Unable to load CUDA cuDNN convolution backward algorithm."))),
        //     _ => algos_bwd[0].algo
        // };
        let algo_bwd = ::cudnn::cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
        let (workspace_bwd, workspace_size_bwd) = match algo_bwd {
            ::cudnn::cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 => (::co::frameworks::cuda::Memory::from_c(0), 0),
            ::cudnn::cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 => (::co::frameworks::cuda::Memory::from_c(0), 0),
            _ => {
                    let workspace_size_bwd = API::get_convolution_backward_data_workspace_size(*CUDNN.id_c(), algo_bwd, *filter_desc.id_c(), *src_desc.id_c(), *conv_desc.id_c(), *dest_desc.id_c()).unwrap();
                    let workspace_backward = ::co::frameworks::cuda::Memory::new(workspace_size_bwd).unwrap();
                    (workspace_backward, workspace_size_bwd)
            }
        };

        Ok(
            ::cudnn::utils::ConvolutionConfig::new(
                algo_fwd, workspace_fwd, workspace_size_fwd,
                algo_bwd, workspace_bwd, workspace_size_bwd,
                conv_desc, filter_desc, filter.remove_copy(self.device()).unwrap().into_cuda().unwrap()
            )
        )
    }

    impl_ops_convolution_for!(f64, Backend<Cuda>);
}

impl_ops_sigmoid_for!(f64, Backend<Cuda>);
impl_ops_relu_for!(f64, Backend<Cuda>);
impl_ops_tanh_for!(f64, Backend<Cuda>);
impl_ops_softmax_for!(f64, Backend<Cuda>);
impl_ops_lrn_for!(f64, Backend<Cuda>);
impl_ops_pooling_for!(f64, Backend<Cuda>);
