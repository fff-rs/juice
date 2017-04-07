//! Provides NN for a CUDA backend.
#![allow(missing_docs)]
use ::plugin::*;
use co::Error as CoError;
use co::prelude::*;
use co::plugin::Error as PluginError;
use co::plugin::numeric_helpers::Float;
use cudnn::*;
use cudnn::utils::ScalParams;


pub use cudnn::utils::DataTypeInfo;

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

impl<T> ICudnnDesc<T> for SharedTensor<T>
    where T: Float + DataTypeInfo,
{
    fn cudnn_tensor_desc(&self) -> Result<TensorDescriptor, PluginError> {
        match TensorDescriptor::new(&self.desc().dims_i32().clone(),
                                    &self.desc().default_stride_i32().clone(),
                                    <T as DataTypeInfo>::cudnn_data_type()) {
            Ok(desc) => Ok(desc),
            Err(_) => {
                Err(PluginError::Plugin("Unable to create CuDNN TensorDescriptor."))
            }
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
            _ => actual_desc
        };
        match TensorDescriptor::new(&override_desc.dims_i32().clone(),
                                    &override_desc.default_stride_i32().clone(),
                                    <T as DataTypeInfo>::cudnn_data_type()) {
            Ok(desc) => Ok(desc),
            Err(_) => {
                Err(PluginError::Plugin("Unable to create CuDNN TensorDescriptor."))
            }
        }
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
        match TensorDescriptor::new(&override_desc.dims_i32().clone(),
                                    &override_desc.default_stride_i32().clone(),
                                    <T as DataTypeInfo>::cudnn_data_type()) {
            Ok(desc) => Ok(desc),
            Err(_) => {
                Err(PluginError::Plugin("Unable to create CuDNN TensorDescriptor."))
            }
        }
    }

    fn cudnn_filter_desc(&self) -> Result<FilterDescriptor, PluginError> {
        match FilterDescriptor::new(&self.desc().dims_i32().clone(),
                                    <T as DataTypeInfo>::cudnn_data_type()) {
            Ok(desc) => Ok(desc),
            Err(_) => {
                Err(PluginError::Plugin("Unable to create CuDNN FilterDescriptor."))
            }
        }
    }

    fn cudnn_convolution_desc(&self, filter: &SharedTensor<T>) -> Result<ConvolutionDescriptor, PluginError> {
        match ConvolutionDescriptor::new(&self.desc().dims_i32().clone(),
                                         &filter.desc().default_stride_i32().clone(),
                                         <T as DataTypeInfo>::cudnn_data_type()) {
            Ok(desc) => Ok(desc),
            Err(_) => {
                Err(PluginError::Plugin("Unable to create CuDNN ConvolutionDescriptor."))
            }
        }
    }
}

impl<T> NN<T> for Backend<Cuda>
    where T: Float + DataTypeInfo,
{
    type CC = utils::ConvolutionConfig;
    type CLRN = utils::NormalizationConfig;
    type CPOOL = utils::PoolingConfig;

    fn init_nn() { let _ = CUDNN.id_c(); }
    fn device(&self) -> &DeviceType { self.device() }
}
impl<'a, T> NNOperationConfig<T> for utils::ConvolutionConfig
    where T: Float + DataTypeInfo,
{ }
impl<T> NNOperationConfig<T> for utils::NormalizationConfig
    where T: Float + DataTypeInfo,
{ }
impl<T> NNOperationConfig<T> for utils::PoolingConfig
    where T: Float + DataTypeInfo,
{ }

impl<T> Sigmoid<T> for Backend<Cuda>
    where T: Float + DataTypeInfo + Default,
{
    fn sigmoid(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>,
        result: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result.add_device(self.device()) { _ => () }

        self.sigmoid_pointwise(x, result)
    }

    fn sigmoid_pointwise(
        &self,
        x: &::co::tensor::SharedTensor<T>,
        result: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.sigmoid_forward(
            &try!(x.cudnn_tensor_desc_flat()), // src_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
            &try!(result.cudnn_tensor_desc_flat()), // dest_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation Sigmoid Forward."))
            }
        }))
    }

    fn sigmoid_grad(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>,
        x_diff: &mut ::co::tensor::SharedTensor<T>,
        result: &mut ::co::tensor::SharedTensor<T>,
        result_diff: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match x_diff.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result_diff.add_device(self.device()) { _ => () }

        self.sigmoid_grad_pointwise(x, x_diff, result, result_diff)
    }

    fn sigmoid_grad_pointwise(
        &self,
        x: &::co::tensor::SharedTensor<T>,
        x_diff: &::co::tensor::SharedTensor<T>,
        result: &::co::tensor::SharedTensor<T>,
        result_diff: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.sigmoid_backward(
            &try!(x.cudnn_tensor_desc_flat()), // src_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
            &try!(x_diff.cudnn_tensor_desc_flat()), // src_diff_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
            &try!(result.cudnn_tensor_desc_flat()), // dest_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(result, self.device()) }), // dest_data
            &try!(result_diff.cudnn_tensor_desc_flat()), // dest_diff_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation Sigmoid Backward."))
            }
        }))
    }
}

impl<T> ConvolutionConfig<T> for ::cudnn::utils::ConvolutionConfig
    where T: Float + DataTypeInfo,
{
    fn workspace_size(&self) -> usize {
        *self.largest_workspace_size()
    }
}

impl<T> Convolution<T> for Backend<Cuda>
    where T: Float + DataTypeInfo,
{
    fn new_convolution_config(&self,
                              src: &SharedTensor<T>,
                              dest: &SharedTensor<T>,
                              filter: &mut SharedTensor<T>,
                              algo_fwd: ConvForwardAlgo,
                              algo_bwd_filter: ConvBackwardFilterAlgo,
                              algo_bwd_data: ConvBackwardDataAlgo,
                              stride: &[i32],
                              zero_padding: &[i32]) ->
        Result<Self::CC, ::co::error::Error>
    {
        let src_desc = try!(src.cudnn_tensor_desc());
        let dest_desc = try!(dest.cudnn_tensor_desc());
        let filter_desc = try!(filter.cudnn_filter_desc());
        let conv_desc = ::cudnn::ConvolutionDescriptor::new(zero_padding, stride,
                                                            <T as DataTypeInfo>::cudnn_data_type()).unwrap();

        let useable_algo_fwd = try!(algo_fwd.find_cudnn_algo(&filter_desc, &conv_desc,
                                                             &src_desc, &dest_desc));
        let useable_algo_bwd_filter = try!(algo_bwd_filter.find_cudnn_algo(&filter_desc, &conv_desc,
                                                                           &src_desc, &dest_desc));
        let useable_algo_bwd_data = try!(algo_bwd_data.find_cudnn_algo(&filter_desc, &conv_desc,
                                                                       &src_desc, &dest_desc));

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
                                                                useable_algo_bwd_filter.as_cudnn().unwrap(),
                                                                *filter_desc.id_c(),
                                                                *conv_desc.id_c(),
                                                                *src_desc.id_c(),
                                                                *dest_desc.id_c())
            .unwrap();
        let mut workspace_size_bwd_data =
            API::get_convolution_backward_data_workspace_size(*CUDNN.id_c(),
                                                              useable_algo_bwd_data.as_cudnn().unwrap(),
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

        Ok(
            ::cudnn::utils::ConvolutionConfig::new(
                useable_algo_fwd.as_cudnn().unwrap(), workspace_size_fwd,
                useable_algo_bwd_filter.as_cudnn().unwrap(), workspace_size_bwd_filter,
                useable_algo_bwd_data.as_cudnn().unwrap(), workspace_size_bwd_data,
                conv_desc, filter_desc
            )
        )
    }
    fn convolution(&self,
                   filter: &mut SharedTensor<T>,
                   x: &mut SharedTensor<T>,
                   result: &mut SharedTensor<T>,
                   workspace: &mut SharedTensor<u8>,
                   config: &Self::CC) -> //::frameworks::cuda::CC
        Result<(), ::co::error::Error>
    {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result.add_device(self.device()) { _ => () }
        match workspace.add_device(self.device()) { _ => () }

        self.convolution_plain(filter, x, result, workspace, config)
    }

    fn convolution_plain(&self,
                         filter: &::co::tensor::SharedTensor<T>,
                         x: &::co::tensor::SharedTensor<T>,
                         result: &mut ::co::tensor::SharedTensor<T>,
                         workspace: &mut ::co::tensor::SharedTensor<u8>,
                         config: &Self::CC) ->
        Result<(), ::co::error::Error>
    {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.convolution_forward(
            config,
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(workspace, self.device()) }),
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(filter, self.device()) }),
            &try!(x.cudnn_tensor_desc()), // src_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
            &try!(result.cudnn_tensor_desc()), // dest_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation convolution Forward."))
            }
        }))
    }

    fn convolution_grad_filter(&self,
                               src_data: &mut SharedTensor<T>,
                               dest_diff: &mut SharedTensor<T>,
                               filter_diff: &mut SharedTensor<T>,
                               workspace: &mut SharedTensor<u8>,
                               config: &Self::CC) ->
        Result<(), ::co::error::Error>
    {
        match src_data.add_device(self.device()) { _ => try!(src_data.sync(self.device())) }
        match dest_diff.add_device(self.device()) { _ => try!(dest_diff.sync(self.device())) }
        match filter_diff.add_device(self.device()) { _ => try!(filter_diff.sync(self.device())) }
        match workspace.add_device(self.device()) { _ => () }

        self.convolution_grad_filter_plain(src_data, dest_diff, filter_diff, workspace, config)
    }

    fn convolution_grad_filter_plain(&self,
                                     src_data: &SharedTensor<T>,
                                     dest_diff: &SharedTensor<T>,
                                     filter_diff: &mut SharedTensor<T>,
                                     workspace: &mut SharedTensor<u8>,
                                     config: &Self::CC) ->
        Result<(), ::co::error::Error>
    {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.convolution_backward_filter(
            config,
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(workspace, self.device()) }),
            &try!(src_data.cudnn_tensor_desc()),
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(src_data, self.device()) }),
            &try!(dest_diff.cudnn_tensor_desc()),
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(dest_diff, self.device()) }),
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(filter_diff, self.device()) }),
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation convolution Backward."))
            }
        }))
    }

    fn convolution_grad_data(&self,
                             filter: &mut SharedTensor<T>,
                             x_diff: &mut SharedTensor<T>,
                             result_diff: &mut SharedTensor<T>,
                             workspace: &mut SharedTensor<u8>,
                             config: &Self::CC) ->
        Result<(), ::co::error::Error>
    {
        match filter.add_device(self.device()) { _ => try!(filter.sync(self.device())) }
        match x_diff.add_device(self.device()) { _ => try!(x_diff.sync(self.device())) }
        match result_diff.add_device(self.device()) { _ => try!(result_diff.sync(self.device())) }
        match workspace.add_device(self.device()) { _ => () }

        self.convolution_grad_data_plain(filter, x_diff, result_diff, workspace, config)
    }

    fn convolution_grad_data_plain(&self,
                                   filter: &SharedTensor<T>,
                                   x_diff: &SharedTensor<T>,
                                   result_diff: &mut SharedTensor<T>,
                                   workspace: &mut SharedTensor<u8>,
                                   config: &Self::CC) ->
        Result<(), ::co::error::Error>
    {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.convolution_backward_data(
            config,
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(workspace, self.device()) }),
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(filter, self.device()) }),
            &try!(x_diff.cudnn_tensor_desc()),
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }),
            &try!(result_diff.cudnn_tensor_desc()),
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }),
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(PluginError::Operation("Unable to execute CUDA cuDNN Activation convolution Backward."))
            }
        }))
    }
}

impl<T> SigmoidPointwise<T> for Backend<Cuda>
    where T: Float + Default + DataTypeInfo,
{
    fn sigmoid_pointwise(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }

        self.sigmoid_pointwise_plain(x)
    }

    fn sigmoid_pointwise_plain(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.sigmoid_forward(
            &try!(x.cudnn_tensor_desc_flat()), // src_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
            &try!(x.cudnn_tensor_desc_flat()), // dest_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(x, self.device()) }), // dest_data
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Sigmoid Pointwise forward."))
            }
        }))
    }

    fn sigmoid_pointwise_grad(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>,
        x_diff: &mut ::co::tensor::SharedTensor<T>,
    ) -> Result<(), ::co::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match x_diff.add_device(self.device()) { _ => try!(x.sync(self.device())) }

        self.sigmoid_pointwise_grad_plain(x, x_diff)
    }

    fn sigmoid_pointwise_grad_plain(
        &self,
        x: &::co::tensor::SharedTensor<T>,
        x_diff: &mut ::co::tensor::SharedTensor<T>,
    ) -> Result<(), ::co::error::Error> {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.sigmoid_backward(
            &try!(x.cudnn_tensor_desc_flat()), // src_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
            &try!(x_diff.cudnn_tensor_desc_flat()), // src_diff_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
            &try!(x.cudnn_tensor_desc_flat()), // dest_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), // dest_data
            &try!(x_diff.cudnn_tensor_desc_flat()), // dest_diff_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(x_diff, self.device()) }), // dest_diff_data
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Sigmoid Pointwise backward."))
            }
        }))
    }
}

impl<T> Relu<T> for Backend<Cuda>
    where T: Float + Default + DataTypeInfo,
{
    fn relu(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>,
        result: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result.add_device(self.device()) { _ => () }

        self.relu_plain(x, result)
    }

    fn relu_plain(
        &self,
        x: &::co::tensor::SharedTensor<T>,
        result: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.relu_forward(
            &try!(x.cudnn_tensor_desc_flat()), // src_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
            &try!(result.cudnn_tensor_desc_flat()), // dest_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation relu Forward."))
            }
        }))
    }

    fn relu_grad(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>,
        x_diff: &mut ::co::tensor::SharedTensor<T>,
        result: &mut ::co::tensor::SharedTensor<T>,
        result_diff: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match x_diff.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result_diff.add_device(self.device()) { _ => () }

        self.relu_grad_plain(x, x_diff, result, result_diff)
    }

    fn relu_grad_plain(
        &self,
        x: &::co::tensor::SharedTensor<T>,
        x_diff: &::co::tensor::SharedTensor<T>,
        result: &::co::tensor::SharedTensor<T>,
        result_diff: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.relu_backward(
            &try!(x.cudnn_tensor_desc_flat()), // src_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
            &try!(x_diff.cudnn_tensor_desc_flat()), // src_diff_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
            &try!(result.cudnn_tensor_desc_flat()), // dest_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(result, self.device()) }), // dest_data
            &try!(result_diff.cudnn_tensor_desc_flat()), // dest_diff_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation relu Backward."))
            }
        }))
    }
}

impl<T> ReluPointwise<T> for Backend<Cuda>
    where T: Float + Default + DataTypeInfo,
{
    fn relu_pointwise(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>,
    ) -> Result<(), ::co::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }

        self.relu_pointwise_plain(x)
    }

    fn relu_pointwise_plain(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>,
    ) -> Result<(), ::co::error::Error> {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.relu_forward(
            &try!(x.cudnn_tensor_desc_flat()), // src_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
            &try!(x.cudnn_tensor_desc_flat()), // dest_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(x, self.device()) }), // dest_data
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN ReLU Pointwise forward."))
            }
        }))
    }

    fn relu_pointwise_grad(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>,
        x_diff: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match x_diff.add_device(self.device()) { _ => try!(x.sync(self.device())) }

        self.relu_pointwise_grad_plain(x, x_diff)
    }

    fn relu_pointwise_grad_plain(
        &self,
        x: &::co::tensor::SharedTensor<T>,
        x_diff: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.relu_backward(
            &try!(x.cudnn_tensor_desc_flat()), // src_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
            &try!(x_diff.cudnn_tensor_desc_flat()), // src_diff_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
            &try!(x.cudnn_tensor_desc_flat()), // dest_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), // dest_data
            &try!(x_diff.cudnn_tensor_desc_flat()), // dest_diff_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(x_diff, self.device()) }), // dest_diff_data
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN ReLU Pointwise backward."))
            }
        }))
    }
}

impl<T> Tanh<T> for Backend<Cuda>
    where T: Float + Default + DataTypeInfo,
{
    fn tanh(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>,
        result: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result.add_device(self.device()) { _ => () }

        self.tanh_plain(x, result)
    }

    fn tanh_plain(
        &self,
        x: &::co::tensor::SharedTensor<T>,
        result: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.tanh_forward(
            &try!(x.cudnn_tensor_desc_flat()), // src_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
            &try!(result.cudnn_tensor_desc_flat()), // dest_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation tanh Forward."))
            }
        }))
    }

    fn tanh_grad(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>,
        x_diff: &mut ::co::tensor::SharedTensor<T>,
        result: &mut ::co::tensor::SharedTensor<T>,
        result_diff: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match x_diff.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result_diff.add_device(self.device()) { _ => () }

        self.tanh_grad_plain(x, x_diff, result, result_diff)
    }

    fn tanh_grad_plain(
        &self,
        x: &::co::tensor::SharedTensor<T>,
        x_diff: &::co::tensor::SharedTensor<T>,
        result: &::co::tensor::SharedTensor<T>,
        result_diff: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.tanh_backward(
            &try!(x.cudnn_tensor_desc_flat()), // src_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
            &try!(x_diff.cudnn_tensor_desc_flat()), // src_diff_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
            &try!(result.cudnn_tensor_desc_flat()), // dest_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(result, self.device()) }), // dest_data
            &try!(result_diff.cudnn_tensor_desc_flat()), // dest_diff_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation tanh Backward."))
            }
        }))
    }
}

impl<T> TanhPointwise<T> for Backend<Cuda>
    where T: Float + Default + DataTypeInfo,
{
    fn tanh_pointwise(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }

        self.tanh_pointwise_plain(x)
    }

    fn tanh_pointwise_plain(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.tanh_forward(
            &try!(x.cudnn_tensor_desc_flat()), // src_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
            &try!(x.cudnn_tensor_desc_flat()), // dest_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(x, self.device()) }), // dest_data
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Tanh Pointwise forward."))
            }
        }))
    }

    fn tanh_pointwise_grad(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>,
        x_diff: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match x_diff.add_device(self.device()) { _ => try!(x.sync(self.device())) }

        self.tanh_pointwise_grad_plain(x, x_diff)
    }

    fn tanh_pointwise_grad_plain(
        &self,
        x: &::co::tensor::SharedTensor<T>,
        x_diff: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.tanh_backward(
            &try!(x.cudnn_tensor_desc_flat()), // src_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
            &try!(x_diff.cudnn_tensor_desc_flat()), // src_diff_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
            &try!(x.cudnn_tensor_desc_flat()), // dest_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), // dest_data
            &try!(x_diff.cudnn_tensor_desc_flat()), // dest_diff_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(x_diff, self.device()) }), // dest_diff_data
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Tanh Pointwise backward."))
            }
        }))
    }
}
impl<T> Softmax<T> for Backend<Cuda>
    where T: Float + Default + DataTypeInfo,
{
    fn softmax(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>,
        result: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result.add_device(self.device()) { _ => () }

        self.softmax_plain(x, result)
    }

    fn softmax_plain(
        &self,
        x: &::co::tensor::SharedTensor<T>,
        result: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.softmax_forward(
            &try!(x.cudnn_tensor_desc_softmax()), // src_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
            &try!(result.cudnn_tensor_desc_softmax()), // dest_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN softmax Forward."))
            }
        }))
    }

    fn softmax_grad(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>,
        x_diff: &mut ::co::tensor::SharedTensor<T>,
        result_diff: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match x_diff.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result_diff.add_device(self.device()) { _ => () }

        self.softmax_grad_plain(x, x_diff, result_diff)
    }

    fn softmax_grad_plain(
        &self,
        x: &::co::tensor::SharedTensor<T>,
        x_diff: &::co::tensor::SharedTensor<T>,
        result_diff: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.softmax_backward(
            &try!(x.cudnn_tensor_desc_softmax()), // src_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
            &try!(x_diff.cudnn_tensor_desc_softmax()), // src_diff_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
            &try!(result_diff.cudnn_tensor_desc_softmax()), // dest_diff_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN softmax Backward."))
            }
        }))
    }
}

impl<T> LogSoftmax<T> for Backend<Cuda>
    where T: Float + Default + DataTypeInfo,
{
    fn log_softmax(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>,
        result: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result.add_device(self.device()) { _ => () }

        self.log_softmax_plain(x, result)
    }

    fn log_softmax_plain(
        &self,
        x: &::co::tensor::SharedTensor<T>,
        result: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.log_softmax_forward(
            &try!(x.cudnn_tensor_desc_softmax()), // src_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
            &try!(result.cudnn_tensor_desc_softmax()), // dest_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN logarithmic softmax Forward."))
            }
        }))
    }

    fn log_softmax_grad(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>,
        x_diff: &mut ::co::tensor::SharedTensor<T>,
        result_diff: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match x_diff.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result_diff.add_device(self.device()) { _ => () }

        self.log_softmax_grad_plain(x, x_diff, result_diff)
    }

    fn log_softmax_grad_plain(
        &self,
        x: &::co::tensor::SharedTensor<T>,
        x_diff: &::co::tensor::SharedTensor<T>,
        result_diff: &mut ::co::tensor::SharedTensor<T>
    ) -> Result<(), ::co::error::Error> {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.log_softmax_backward(
            &try!(x.cudnn_tensor_desc_softmax()), // src_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
            &try!(x_diff.cudnn_tensor_desc_softmax()), // src_diff_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
            &try!(result_diff.cudnn_tensor_desc_softmax()), // dest_diff_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN logarithmic softmax Backward."))
            }
        }))
    }
}

impl<T> LRN<T> for Backend<Cuda>
    where T: Float + Default + DataTypeInfo,
{
    fn new_lrn_config(
        &self,
        n: u32,
        alpha: f64,
        beta: f64,
        k: f64
    ) -> Result<Self::CLRN, ::co::error::Error> {
        Ok(CUDNN.init_normalization(n, alpha, beta, k).unwrap())
    }

    fn lrn(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>,
        result: &mut ::co::tensor::SharedTensor<T>,
        config: &Self::CLRN //::frameworks::cuda::CC
    ) -> Result<(), ::co::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result.add_device(self.device()) { _ => () }

        self.lrn_plain(x, result, config)
    }

    fn lrn_plain(
        &self,
        x: &::co::tensor::SharedTensor<T>,
        result: &mut ::co::tensor::SharedTensor<T>,
        config: &Self::CLRN
    ) -> Result<(), ::co::error::Error> {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.lrn_forward(
            config,
            &try!(x.cudnn_tensor_desc()), // src_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
            &try!(result.cudnn_tensor_desc()), // dest_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation lrn Forward."))
            }
        }))
    }

    #[allow(unused_variables)]
    fn lrn_grad(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>,
        x_diff: &mut ::co::tensor::SharedTensor<T>,
        result: &mut ::co::tensor::SharedTensor<T>,
        result_diff: &mut ::co::tensor::SharedTensor<T>,
        config: &Self::CLRN
    ) -> Result<(), ::co::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match x_diff.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result_diff.add_device(self.device()) { _ => () }

        self.lrn_grad_plain(x, x_diff, result, result_diff, config)
    }

    #[allow(unused_variables)]
    fn lrn_grad_plain(
        &self,
        x: &::co::tensor::SharedTensor<T>,
        x_diff: &::co::tensor::SharedTensor<T>,
        result: &::co::tensor::SharedTensor<T>,
        result_diff: &mut ::co::tensor::SharedTensor<T>,
        config: &Self::CLRN
    ) -> Result<(), ::co::error::Error> {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.lrn_backward(
            config,
            &try!(x.cudnn_tensor_desc()), // src_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
            &try!(x_diff.cudnn_tensor_desc()), // src_diff_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
            &try!(result.cudnn_tensor_desc()), // dest_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(result, self.device()) }), // dest_data
            &try!(result_diff.cudnn_tensor_desc()), // dest_diff_desc
            try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation lrn Backward."))
            }
        }))
    }
}

impl<T> Pooling<T> for Backend<Cuda>
    where T: Float + Default + DataTypeInfo,
{
    fn new_pooling_config(
        &self,
        window: &[i32],
        padding: &[i32],
        stride: &[i32],
    ) -> Result<Self::CPOOL, ::co::error::Error> {
        let pooling_avg = ::cudnn::PoolingDescriptor::new(::cudnn::cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, window, padding, stride).unwrap();
        let pooling_max = ::cudnn::PoolingDescriptor::new(::cudnn::cudnnPoolingMode_t::CUDNN_POOLING_MAX, window, padding, stride).unwrap();
        Ok(::cudnn::utils::PoolingConfig::new(pooling_avg, pooling_max))
    }

    fn pooling_max(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>,
        result: &mut ::co::tensor::SharedTensor<T>,
        config: &Self::CPOOL
    ) -> Result<(), ::co::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result.add_device(self.device()) { _ => () }

        self.pooling_max_pointwise(x, result, config)
    }

    fn pooling_max_pointwise(
        &self,
        x: &::co::tensor::SharedTensor<T>,
        result: &mut ::co::tensor::SharedTensor<T>,
        config: &Self::CPOOL
    ) -> Result<(), ::co::error::Error> {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.pooling_max_forward(
            config,
            &try!(x.cudnn_tensor_desc()), // src_desc
            read!(x, self), //src_data
            &try!(result.cudnn_tensor_desc()), // dest_desc
            write_only!(result, self), // dest_data
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation pooling Forward."))
            }
        }))
    }

    #[allow(unused_variables)]
    fn pooling_max_grad(
        &self,
        x: &mut ::co::tensor::SharedTensor<T>,
        x_diff: &mut ::co::tensor::SharedTensor<T>,
        result: &mut ::co::tensor::SharedTensor<T>,
        result_diff: &mut ::co::tensor::SharedTensor<T>,
        config: &Self::CPOOL
    ) -> Result<(), ::co::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match x_diff.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result_diff.add_device(self.device()) { _ => () }

        self.pooling_max_grad_pointwise(x, x_diff, result, result_diff, config)
    }

    #[allow(unused_variables)]
    fn pooling_max_grad_pointwise(
        &self,
        x: &::co::tensor::SharedTensor<T>,
        x_diff: &::co::tensor::SharedTensor<T>,
        result: &::co::tensor::SharedTensor<T>,
        result_diff: &mut ::co::tensor::SharedTensor<T>,
        config: &Self::CPOOL
    ) -> Result<(), ::co::error::Error> {
        let scal_params: ::cudnn::utils::ScalParams<T> = ::cudnn::utils::ScalParams::default();

        Ok(try!(match CUDNN.pooling_max_backward(
            config,
            &try!(x.cudnn_tensor_desc()), // src_desc
            read!(x, self), //src_data
            &try!(x_diff.cudnn_tensor_desc()), // src_diff_desc
            read!(x_diff, self), //src_diff_data
            &try!(result.cudnn_tensor_desc()), // dest_desc
            write_only!(result, self), // dest_data
            &try!(result_diff.cudnn_tensor_desc()), // dest_diff_desc
            write_only!(result_diff, self), // dest_diff_data
            scal_params
        ) {
            Ok(_) => Ok(()),
            Err(_) => {
                Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation pooling Backward."))
            }
        }))
    }
}
