//! Provides NN for a CUDA backend.
#![allow(missing_docs)]

pub use crate::cudnn::utils::{DataType, DataTypeInfo};
use crate::cudnn::*;
use crate::plugin::*;
use co::plugin::numeric_helpers::Float;
use co::plugin::Error as PluginError;
use co::prelude::*;
use co::Error;
use coaster as co;

#[macro_use]
pub mod helper;

pub(crate) fn rnn_sequence_descriptors(
    sequence_length: i32,
    input_size: i32,
    hidden_size: i32,
    batch_size: i32,
    num_layers: i32,
    direction_mode: DirectionMode,
    data_type: DataType,
) -> Result<RnnSequenceDescriptors, Error> {
    let bidirectional = if direction_mode == DirectionMode::UniDirectional {
        1
    } else {
        2 // bidirection needs twice as much memory
    };

    // Treating the input split by batch then input like in a typical NCHW cell.
    let dim_input = vec![num_layers, batch_size, input_size];
    let dim_output = vec![num_layers, batch_size, hidden_size];
    let dim_hidden_cell = vec![num_layers * bidirectional, batch_size, hidden_size];
    let _stride_input = vec![dim_input[2] * dim_input[1], dim_input[2], 1];
    let _stride_output = vec![dim_output[2] * dim_output[1], dim_output[2], 1];
    let stride_hidden_cell = vec![
        dim_hidden_cell[2] * dim_hidden_cell[1],
        dim_hidden_cell[2],
        1,
    ];

    let mut x_desc: Vec<TensorDescriptor> = Vec::with_capacity(sequence_length as usize);
    let mut y_desc: Vec<TensorDescriptor> = Vec::with_capacity(sequence_length as usize);
    let mut dx_desc: Vec<TensorDescriptor> = Vec::with_capacity(sequence_length as usize);
    let mut dy_desc: Vec<TensorDescriptor> = Vec::with_capacity(sequence_length as usize);

    {
        let dim_x = vec![batch_size, input_size, 1];
        let stride_x = vec![dim_x[2] * dim_x[1], dim_x[2], 1];
        let dim_y = vec![batch_size, hidden_size * bidirectional, 1];
        let stride_y = vec![dim_y[2] * dim_y[1], dim_y[2], 1];
        for _ in 0..sequence_length {
            x_desc.push(TensorDescriptor::new(&dim_x, &stride_x, data_type).unwrap());
            dx_desc.push(TensorDescriptor::new(&dim_x, &stride_x, data_type).unwrap());
            y_desc.push(TensorDescriptor::new(&dim_y, &stride_y, data_type).unwrap());
            dy_desc.push(TensorDescriptor::new(&dim_y, &stride_y, data_type).unwrap());
        }
    }

    Ok(RnnSequenceDescriptors {
        x_desc,
        y_desc,
        dx_desc,
        dy_desc,
        hx_desc: TensorDescriptor::new(&dim_hidden_cell, &stride_hidden_cell, data_type).unwrap(),
        hy_desc: TensorDescriptor::new(&dim_hidden_cell, &stride_hidden_cell, data_type).unwrap(),
        cx_desc: TensorDescriptor::new(&dim_hidden_cell, &stride_hidden_cell, data_type).unwrap(),
        cy_desc: TensorDescriptor::new(&dim_hidden_cell, &stride_hidden_cell, data_type).unwrap(),
        dhx_desc: TensorDescriptor::new(&dim_hidden_cell, &stride_hidden_cell, data_type).unwrap(),
        dhy_desc: TensorDescriptor::new(&dim_hidden_cell, &stride_hidden_cell, data_type).unwrap(),
        dcx_desc: TensorDescriptor::new(&dim_hidden_cell, &stride_hidden_cell, data_type).unwrap(),
        dcy_desc: TensorDescriptor::new(&dim_hidden_cell, &stride_hidden_cell, data_type).unwrap(),
    })
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

    fn cudnn_convolution_desc(
        &self,
        filter: &SharedTensor<T>,
    ) -> Result<ConvolutionDescriptor, PluginError>;

    fn cudnn_rnn_desc(
        &self,
        cudnn_framework: &Cudnn,
        hidden_size: i32,
        num_layers: i32,
        dropout_desc: utils::DropoutConfig,
        input_mode: cudnnRNNInputMode_t,
        direction: cudnnDirectionMode_t,
        mode: cudnnRNNMode_t,
        algorithm: cudnnRNNAlgo_t,
        padding_mode: cudnnRNNPaddingMode_t,
    ) -> Result<RnnDescriptor, PluginError>;
}

impl ConvForwardAlgo {
    /// Tries to return the matching cuDNN type for the enum value.
    fn as_cudnn(&self) -> Result<cudnnConvolutionFwdAlgo_t, Error> {
        use crate::cudnn::cudnnConvolutionFwdAlgo_t::*;
        use crate::ConvForwardAlgo::*;
        Ok(match *self {
            Auto => {
                return Err(Error::Plugin(PluginError::Plugin(
                    "Can't create cuDNN convolution forward algorithm from \
                 ConvForwardAlgo::Auto. Use `find_cudnn_algo` to find an algorithm.",
                )))
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
        use crate::cudnn::cudnnConvolutionFwdAlgo_t::*;
        use crate::ConvForwardAlgo::*;
        match *algo {
            CUDNN_CONVOLUTION_FWD_ALGO_GEMM => GEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM => ImplicitGEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM => ImplicitPrecompiledGEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_FFT => FFT,
            CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING => FFTTiling,
            CUDNN_CONVOLUTION_FWD_ALGO_DIRECT => Direct,
            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD => Winograd,
            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED => WinogradNonFused,
            _ => unreachable!(),
        }
    }

    /// Try to find best algorithm for a operation that uses the provided descriptors.
    fn find_cudnn_algo(
        &self,
        cudnn_framework: &Cudnn,
        filter_desc: &FilterDescriptor,
        conv_desc: &ConvolutionDescriptor,
        src_desc: &TensorDescriptor,
        dest_desc: &TensorDescriptor,
    ) -> Result<ConvForwardAlgo, Error> {
        if !self.is_auto() {
            return Ok(*self);
        }
        let algos = API::find_convolution_forward_algorithm(
            *cudnn_framework.id_c(),
            *filter_desc.id_c(),
            *conv_desc.id_c(),
            *src_desc.id_c(),
            *dest_desc.id_c(),
        )
        .unwrap();
        let algo = match algos.len() {
            0 => {
                return Err(Error::Plugin(PluginError::Operation(
                    "Unable to find CUDA cuDNN convolution forward algorithm.",
                )))
            }
            _ => algos[0].algo,
        };
        Ok(ConvForwardAlgo::from_cudnn(&algo))
    }
}

impl ConvBackwardFilterAlgo {
    /// Tries to return the matching cuDNN type for the enum value.
    fn as_cudnn(&self) -> Result<cudnnConvolutionBwdFilterAlgo_t, Error> {
        use crate::cudnn::cudnnConvolutionBwdFilterAlgo_t::*;
        use crate::ConvBackwardFilterAlgo::*;
        Ok(match *self {
            Auto => {
                return Err(Error::Plugin(PluginError::Plugin(
                    "Can't create cuDNN convolution backward filter algorithm from \
                 ConvBackwardFilterAlgo::Auto. Use `find_cudnn_algo` to find an \
                 algorithm.",
                )))
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
        use crate::cudnn::cudnnConvolutionBwdFilterAlgo_t::*;
        use crate::ConvBackwardFilterAlgo::*;
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
    fn find_cudnn_algo(
        &self,
        cudnn_framework: &Cudnn,
        filter_desc: &FilterDescriptor,
        conv_desc: &ConvolutionDescriptor,
        src_desc: &TensorDescriptor,
        dest_desc: &TensorDescriptor,
    ) -> Result<ConvBackwardFilterAlgo, Error> {
        if !self.is_auto() {
            return Ok(*self);
        }
        let algos = API::find_convolution_backward_filter_algorithm(
            *cudnn_framework.id_c(),
            *filter_desc.id_c(),
            *conv_desc.id_c(),
            *src_desc.id_c(),
            *dest_desc.id_c(),
        )
        .unwrap();
        let algo = match algos.len() {
            0 => {
                return Err(Error::Plugin(PluginError::Operation(
                    "Unable to find CUDA cuDNN convolution backward filter algorithm.",
                )))
            }
            _ => algos[0].algo,
        };
        Ok(ConvBackwardFilterAlgo::from_cudnn(&algo))
    }
}

impl ConvBackwardDataAlgo {
    /// Tries to return the matching cuDNN type for the enum value.
    fn as_cudnn(&self) -> Result<cudnnConvolutionBwdDataAlgo_t, Error> {
        use crate::cudnn::cudnnConvolutionBwdDataAlgo_t::*;
        use crate::ConvBackwardDataAlgo::*;
        Ok(match *self {
            Auto => {
                return Err(Error::Plugin(PluginError::Plugin(
                    "Can't create cuDNN convolution backward data algorithm from \
                 ConvBackwardDataAlgo::Auto. Use `find_cudnn_algo` to find \
                 an algorithm.",
                )))
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
        use crate::cudnn::cudnnConvolutionBwdDataAlgo_t::*;
        use crate::ConvBackwardDataAlgo::*;
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
    fn find_cudnn_algo(
        &self,
        cudnn_framework: &Cudnn,
        filter_desc: &FilterDescriptor,
        conv_desc: &ConvolutionDescriptor,
        src_desc: &TensorDescriptor,
        dest_desc: &TensorDescriptor,
    ) -> Result<ConvBackwardDataAlgo, Error> {
        if !self.is_auto() {
            return Ok(*self);
        }
        let algos = API::find_convolution_backward_data_algorithm(
            *cudnn_framework.id_c(),
            *filter_desc.id_c(),
            *conv_desc.id_c(),
            *src_desc.id_c(),
            *dest_desc.id_c(),
        )
        .unwrap();

        let algo = match algos.len() {
            0 => {
                return Err(Error::Plugin(PluginError::Operation(
                    "Unable to find CUDA cuDNN convolution backward data algorithm.",
                )))
            }
            _ => algos[0].algo,
        };
        Ok(ConvBackwardDataAlgo::from_cudnn(&algo))
    }
}

impl<T> ICudnnDesc<T> for SharedTensor<T>
where
    T: Float + DataTypeInfo,
{
    fn cudnn_tensor_desc(&self) -> Result<TensorDescriptor, PluginError> {
        exec!(TensorDescriptor::new(
            &self.desc().dims_i32().clone(),
            &self.desc().default_stride_i32().clone(),
            <T as DataTypeInfo>::cudnn_data_type(),
        ) => "Unable to create CuDNN TensorDescriptor.")
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
        exec!(TensorDescriptor::new(
            &override_desc.dims_i32().clone(),
            &override_desc.default_stride_i32().clone(),
            <T as DataTypeInfo>::cudnn_data_type(),
        ) => "Unable to create CuDNN TensorDescriptor.")
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
        exec!(TensorDescriptor::new(
            &override_desc.dims_i32().clone(),
            &override_desc.default_stride_i32().clone(),
            <T as DataTypeInfo>::cudnn_data_type(),
        ) => "Unable to create CuDNN TensorDescriptor.")
    }

    fn cudnn_filter_desc(&self) -> Result<FilterDescriptor, PluginError> {
        exec!(FilterDescriptor::new(
            &self.desc().dims_i32().clone(),
            <T as DataTypeInfo>::cudnn_data_type(),
        ) => "Unable to create CuDNN FilterDescriptor.")
    }

    //fn cudnn_tensor_desc_rnn(&self) -> Result<TensorDescriptor, PluginError> {
    //    let actual_desc : Vec<usize> = self.desc().clone();
    //    unimplemented!()
    //}

    fn cudnn_convolution_desc(
        &self,
        filter: &SharedTensor<T>,
    ) -> Result<ConvolutionDescriptor, PluginError> {
        exec!(ConvolutionDescriptor::new(
            &self.desc().dims_i32().clone(),
            &filter.desc().default_stride_i32().clone(),
            <T as DataTypeInfo>::cudnn_data_type(),
        ) => "Unable to create CuDNN ConvolutionDescriptor.")
    }

    fn cudnn_rnn_desc(
        &self,
        cudnn_framework: &Cudnn,
        hidden_size: i32,
        num_layers: i32,
        dropout_desc: utils::DropoutConfig,
        input_mode: cudnnRNNInputMode_t,
        direction: cudnnDirectionMode_t,
        mode: cudnnRNNMode_t,
        algorithm: cudnnRNNAlgo_t,
        padding_mode: cudnnRNNPaddingMode_t,
    ) -> Result<RnnDescriptor, PluginError> {
        exec!(RnnDescriptor::new(
            &cudnn_framework,
            hidden_size,
            num_layers,
            dropout_desc,
            input_mode,
            direction,
            mode,
            algorithm,
            <T as DataTypeInfo>::cudnn_data_type(),
            padding_mode,
        ) => "Unable to create CuDNN RNNDescriptor")
    }
}

impl<T> NN<T> for Backend<Cuda>
where
    T: Float + DataTypeInfo,
{
    type CC = utils::ConvolutionContext;
    type CLRN = utils::NormalizationConfig;
    type CPOOL = utils::PoolingConfig;
    type CDROP = utils::DropoutConfig;
    type CRNN = utils::RnnConfig;

    fn init_nn() {
        //let _ = cudnn_framework.id_c();
    }
}

impl<'a, T> NNOperationConfig<T> for utils::ConvolutionContext where T: Float + DataTypeInfo {}
impl<T> NNOperationConfig<T> for utils::RnnConfig where T: Float + DataTypeInfo {}
impl<T> NNOperationConfig<T> for utils::NormalizationConfig where T: Float + DataTypeInfo {}
impl<T> NNOperationConfig<T> for utils::PoolingConfig where T: Float + DataTypeInfo {}
impl<T> NNOperationConfig<T> for utils::DropoutConfig where T: Float + DataTypeInfo {}

impl<T> Sigmoid<T> for Backend<Cuda>
where
    T: Float + DataTypeInfo + Default,
{
    fn sigmoid(&self, x: &SharedTensor<T>, result: &mut SharedTensor<T>) -> Result<(), Error> {
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let cudnn_framework = self.framework().cudnn();
        let r_desc = result.cudnn_tensor_desc_flat()?;
        let x_mem = read!(x, self);
        let r_mem = write_only!(result, self);

        exec2!(cudnn_framework.sigmoid_forward(
            &cudnn_framework.init_activation().unwrap(),
            &x.cudnn_tensor_desc_flat()?,
            trans!(x_mem),
            &r_desc,
            trans_mut!(r_mem),
            scal_params,
        ) => "Unable to execute CUDA cuDNN Activation Sigmoid Forward.")
    }

    fn sigmoid_grad(
        &self,
        x: &SharedTensor<T>,
        x_diff: &SharedTensor<T>,
        result: &SharedTensor<T>,
        result_diff: &mut SharedTensor<T>,
    ) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let dr_desc = result_diff.cudnn_tensor_desc_flat()?;
        let x_mem = read!(x, self);
        let dx_mem = read!(x_diff, self);
        let r_mem = read!(result, self);
        let dr_mem = write_only!(result_diff, self);
        exec2!(cudnn_framework.sigmoid_backward(
            &cudnn_framework.init_activation().unwrap(),
            &x.cudnn_tensor_desc_flat()?,
            trans!(x_mem),
            &x_diff.cudnn_tensor_desc_flat()?,
            trans!(dx_mem),
            &result.cudnn_tensor_desc_flat()?,
            trans!(r_mem),
            &dr_desc,
            trans_mut!(dr_mem),
            scal_params,
        ) => "Unable to execute CUDA cuDNN Activation Sigmoid Backward.")
    }
}

impl<T> ConvolutionContext<T> for crate::cudnn::utils::ConvolutionContext where
    T: Float + DataTypeInfo
{
}

impl<T> Convolution<T> for Backend<Cuda>
where
    T: Float + DataTypeInfo,
{
    fn new_convolution_context(
        &self,
        src: &SharedTensor<T>,
        dest: &SharedTensor<T>,
        filter: &SharedTensor<T>,
        algo_fwd: ConvForwardAlgo,
        algo_bwd_filter: ConvBackwardFilterAlgo,
        algo_bwd_data: ConvBackwardDataAlgo,
        stride: &[i32],
        zero_padding: &[i32],
    ) -> Result<Self::CC, Error> {
        let cudnn_framework = self.framework().cudnn();
        let src_desc = src.cudnn_tensor_desc()?;
        let dest_desc = dest.cudnn_tensor_desc()?;
        let filter_desc = filter.cudnn_filter_desc()?;
        let conv_desc = crate::cudnn::ConvolutionDescriptor::new(
            zero_padding,
            stride,
            <T as DataTypeInfo>::cudnn_data_type(),
        )
        .unwrap();

        let useable_algo_fwd = algo_fwd.find_cudnn_algo(
            cudnn_framework,
            &filter_desc,
            &conv_desc,
            &src_desc,
            &dest_desc,
        )?;
        let useable_algo_bwd_filter = algo_bwd_filter.find_cudnn_algo(
            cudnn_framework,
            &filter_desc,
            &conv_desc,
            &src_desc,
            &dest_desc,
        )?;
        let useable_algo_bwd_data = algo_bwd_data.find_cudnn_algo(
            cudnn_framework,
            &filter_desc,
            &conv_desc,
            &src_desc,
            &dest_desc,
        )?;

        let mut workspace_size_fwd = API::get_convolution_forward_workspace_size(
            *cudnn_framework.id_c(),
            useable_algo_fwd.as_cudnn().unwrap(),
            *filter_desc.id_c(),
            *conv_desc.id_c(),
            *src_desc.id_c(),
            *dest_desc.id_c(),
        )
        .unwrap();
        let mut workspace_size_bwd_filter = API::get_convolution_backward_filter_workspace_size(
            *cudnn_framework.id_c(),
            useable_algo_bwd_filter.as_cudnn().unwrap(),
            *filter_desc.id_c(),
            *conv_desc.id_c(),
            *src_desc.id_c(),
            *dest_desc.id_c(),
        )
        .unwrap();
        let mut workspace_size_bwd_data = API::get_convolution_backward_data_workspace_size(
            *cudnn_framework.id_c(),
            useable_algo_bwd_data.as_cudnn().unwrap(),
            *filter_desc.id_c(),
            *conv_desc.id_c(),
            *src_desc.id_c(),
            *dest_desc.id_c(),
        )
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

        Ok(crate::cudnn::utils::ConvolutionContext::new(
            useable_algo_fwd.as_cudnn().unwrap(),
            workspace_size_fwd,
            useable_algo_bwd_filter.as_cudnn().unwrap(),
            workspace_size_bwd_filter,
            useable_algo_bwd_data.as_cudnn().unwrap(),
            workspace_size_bwd_data,
            conv_desc,
            filter_desc,
        ))
    }

    fn convolution(
        &self,
        filter: &SharedTensor<T>,
        x: &SharedTensor<T>,
        result: &mut SharedTensor<T>,
        context: &mut Self::CC,
    ) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();

        let r_desc = result.cudnn_tensor_desc()?;
        let f_mem = read!(filter, self);
        let x_mem = read!(x, self);
        let r_mem = write_only!(result, self);

        exec2!(cudnn_framework.convolution_forward(
            context,
            trans!(f_mem),
            &x.cudnn_tensor_desc()?, // src_desc
            trans!(x_mem),
            &r_desc,
            trans_mut!(r_mem),
            scal_params,
        ) => "Unable to execute CUDA cuDNN Activation convolution Forward.")
    }

    fn convolution_grad_filter(
        &self,
        src_data: &SharedTensor<T>,
        dest_diff: &SharedTensor<T>,
        filter_diff: &mut SharedTensor<T>,
        context: &mut Self::CC,
    ) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let s_mem = read!(src_data, self);
        let dd_mem = read!(dest_diff, self);
        let df_mem = write_only!(filter_diff, self);
        exec2!(cudnn_framework.convolution_backward_filter(
            context,
            &src_data.cudnn_tensor_desc()?,
            trans!(s_mem),
            &dest_diff.cudnn_tensor_desc()?,
            trans!(dd_mem),
            trans_mut!(df_mem),
            scal_params,
        )  => "Unable to execute CUDA cuDNN Activation convolution Backward.")
    }

    fn convolution_grad_data(
        &self,
        filter: &SharedTensor<T>,
        x_diff: &SharedTensor<T>,
        result_diff: &mut SharedTensor<T>,
        context: &mut Self::CC,
    ) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();

        let dr_desc = result_diff.cudnn_tensor_desc()?;
        let f_mem = read!(filter, self);
        let dx_mem = read!(x_diff, self);
        let dr_mem = write_only!(result_diff, self);
        exec2!(cudnn_framework.convolution_backward_data(
            context,
            trans!(f_mem),
            &x_diff.cudnn_tensor_desc()?,
            trans!(dx_mem),
            &dr_desc,
            trans_mut!(dr_mem),
            scal_params,
        )  => "Unable to execute CUDA cuDNN Activation convolution Backward.")
    }
}

impl<T> RnnConfig<T> for crate::cudnn::utils::RnnConfig where T: Float + DataTypeInfo {}

impl RnnInputMode {
    fn as_cudnn(&self) -> Result<cudnnRNNInputMode_t, Error> {
        Ok(match self {
            RnnInputMode::LinearInput => cudnnRNNInputMode_t::CUDNN_LINEAR_INPUT,
            RnnInputMode::SkipInput => cudnnRNNInputMode_t::CUDNN_SKIP_INPUT,
        })
    }

    fn from_cudnn(input: cudnnRNNInputMode_t) -> Self {
        match input {
            cudnnRNNInputMode_t::CUDNN_LINEAR_INPUT => RnnInputMode::LinearInput,
            cudnnRNNInputMode_t::CUDNN_SKIP_INPUT => RnnInputMode::SkipInput,
            _ => unreachable!(),
        }
    }
}

impl DirectionMode {
    fn as_cudnn(&self) -> Result<cudnnDirectionMode_t, Error> {
        Ok(match self {
            DirectionMode::BiDirectional => cudnnDirectionMode_t::CUDNN_BIDIRECTIONAL,
            DirectionMode::UniDirectional => cudnnDirectionMode_t::CUDNN_UNIDIRECTIONAL,
        })
    }

    fn from_cudnn(direction: cudnnDirectionMode_t) -> Self {
        match direction {
            cudnnDirectionMode_t::CUDNN_BIDIRECTIONAL => DirectionMode::BiDirectional,
            cudnnDirectionMode_t::CUDNN_UNIDIRECTIONAL => DirectionMode::UniDirectional,
            _ => unreachable!(),
        }
    }
}

impl RnnNetworkMode {
    fn as_cudnn(&self) -> Result<cudnnRNNMode_t, Error> {
        Ok(match self {
            RnnNetworkMode::ReLU => cudnnRNNMode_t::CUDNN_RNN_RELU,
            RnnNetworkMode::Tanh => cudnnRNNMode_t::CUDNN_RNN_TANH,
            RnnNetworkMode::LSTM => cudnnRNNMode_t::CUDNN_LSTM,
            RnnNetworkMode::GRU => cudnnRNNMode_t::CUDNN_GRU,
        })
    }

    fn from_cudnn(network_mode: cudnnRNNMode_t) -> Self {
        match network_mode {
            cudnnRNNMode_t::CUDNN_RNN_RELU => RnnNetworkMode::ReLU,
            cudnnRNNMode_t::CUDNN_RNN_TANH => RnnNetworkMode::Tanh,
            cudnnRNNMode_t::CUDNN_LSTM => RnnNetworkMode::LSTM,
            cudnnRNNMode_t::CUDNN_GRU => RnnNetworkMode::GRU,
            _ => unreachable!(),
        }
    }
}

impl RnnAlgorithm {
    fn as_cudnn(&self) -> Result<cudnnRNNAlgo_t, Error> {
        Ok(match self {
            RnnAlgorithm::PersistDynamic => cudnnRNNAlgo_t::CUDNN_RNN_ALGO_PERSIST_DYNAMIC,
            RnnAlgorithm::PersistStatic => cudnnRNNAlgo_t::CUDNN_RNN_ALGO_PERSIST_STATIC,
            RnnAlgorithm::Standard => cudnnRNNAlgo_t::CUDNN_RNN_ALGO_STANDARD,
            RnnAlgorithm::Count => cudnnRNNAlgo_t::CUDNN_RNN_ALGO_COUNT,
        })
    }

    fn from_cudnn(algorithm: cudnnRNNAlgo_t) -> Self {
        match algorithm {
            cudnnRNNAlgo_t::CUDNN_RNN_ALGO_PERSIST_DYNAMIC => RnnAlgorithm::PersistDynamic,
            cudnnRNNAlgo_t::CUDNN_RNN_ALGO_PERSIST_STATIC => RnnAlgorithm::PersistStatic,
            cudnnRNNAlgo_t::CUDNN_RNN_ALGO_STANDARD => RnnAlgorithm::Standard,
            cudnnRNNAlgo_t::CUDNN_RNN_ALGO_COUNT => RnnAlgorithm::Count,
            _ => unreachable!(),
        }
    }
}

impl MathType {
    fn as_cudnn(&self) -> Result<cudnnMathType_t, Error> {
        match self {
            MathType::Default => Ok(cudnnMathType_t::CUDNN_DEFAULT_MATH),
            MathType::TensorOPMath => Ok(cudnnMathType_t::CUDNN_TENSOR_OP_MATH),
            MathType::TensorOPMathAllowConversion => {
                Ok(cudnnMathType_t::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)
            }
        }
    }

    fn from_cudnn(math_type: cudnnMathType_t) -> MathType {
        match math_type {
            cudnnMathType_t::CUDNN_DEFAULT_MATH => MathType::Default,
            cudnnMathType_t::CUDNN_TENSOR_OP_MATH => MathType::TensorOPMath,
            cudnnMathType_t::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION => {
                MathType::TensorOPMathAllowConversion
            }
            _ => unreachable!(),
        }
    }
}

impl RnnPaddingMode {
    fn as_cudnn(&self) -> Result<cudnnRNNPaddingMode_t, Error> {
        match self {
            RnnPaddingMode::Enabled => Ok(CUDNN_RNN_PADDED_IO_ENABLED),
            RnnPaddingMode::Disabled => Ok(CUDNN_RNN_PADDED_IO_DISABLED),
        }
    }

    fn from_cudnn(padding_type: cudnnRNNPaddingMode_t) -> RnnPaddingMode {
        match padding_type {
            CUDNN_RNN_PADDED_IO_ENABLED => RnnPaddingMode::Enabled,
            CUDNN_RNN_PADDED_IO_DISABLED => RnnPaddingMode::Disabled,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
// All RNN Sequence Descriptors are generated on a single pass in CUDNN example code
// As such, defining them all in one function appears to be the simplest method of reproducing
// this work in Rust, but passing back a tuple is unwieldy as the tuple grows beyond 2 - 3 values.
/// Struct to hold all Sequence Descriptors for an RNN Pass
pub struct RnnSequenceDescriptors {
    /// Input Descriptor
    pub x_desc: Vec<TensorDescriptor>,
    /// Output Descriptor
    pub y_desc: Vec<TensorDescriptor>,
    /// Gradient Input Descriptor
    pub dx_desc: Vec<TensorDescriptor>,
    /// Gradient Output Descriptor
    pub dy_desc: Vec<TensorDescriptor>,
    /// Hidden Input Descriptor
    pub hx_desc: TensorDescriptor,
    /// Cell Input Descriptor
    pub cx_desc: TensorDescriptor,
    /// Hidden Output Descriptor
    pub hy_desc: TensorDescriptor,
    /// Cell Output Descriptor
    pub cy_desc: TensorDescriptor,
    /// Gradient Hidden Input Descriptor
    pub dhx_desc: TensorDescriptor,
    /// Gradient Cell Input Descriptor
    pub dcx_desc: TensorDescriptor,
    /// Gradient Hidden Output Descriptor
    pub dhy_desc: TensorDescriptor,
    /// Gradient Cell Output Descriptor
    pub dcy_desc: TensorDescriptor,
}

impl<T> Rnn<T> for Backend<Cuda>
where
    T: Float + DataTypeInfo,
{
    fn generate_rnn_weight_description(
        &self,
        rnn_config: &Self::CRNN,
        input_size: i32,
    ) -> Result<Vec<usize>, Error> {
        let cudnn_framework = self.framework().cudnn();
        let data_type = <T as DataTypeInfo>::cudnn_data_type();

        // According to cuDNN API reference and examples, xDesc should have a
        // least 3 dimensions with batch_size being the first. However, weights
        // size does not depend on batch size and we'd like to avoid having to
        // specify batch size in advance (as it can change during execution).
        // So we use batch_size = 1 as it appers to work well.
        let dim_x = vec![1, input_size, 1];
        let stride_x = vec![dim_x[2] * dim_x[1], dim_x[2], 1];

        // dummy desc to get the param size
        let x_desc = TensorDescriptor::new(&dim_x, &stride_x, data_type).unwrap();

        let weight_size: usize = exec2!(API::get_rnn_params_size(
            *cudnn_framework.id_c(),
            *rnn_config.rnn_desc.id_c(),
            // Input. A fully packed tensor descriptor describing the input to one recurrent iteration.
            // Appears to be a single descriptor, not an array of tensor descriptors.
            *x_desc.id_c(),
            data_type,
        ) => "Unable to get CudNN Rnn Params Size.")?;

        // TODO: Update for different sizing.
        let dim_w: Vec<usize> = vec![weight_size / <T as DataTypeInfo>::size(), 1, 1];
        Ok(dim_w)
    }

    fn new_rnn_config(
        &self,
        dropout_probability: Option<f32>,
        dropout_seed: Option<u64>,
        sequence_length: i32,
        network_mode: RnnNetworkMode,
        input_mode: RnnInputMode,
        direction_mode: DirectionMode,
        algorithm: RnnAlgorithm,
        input_size: i32,
        hidden_size: i32,
        num_layers: i32,
    ) -> Result<Self::CRNN, Error> {
        // Use batch size of 1 to initialize workspace. It will be resized in
        // rnn_forward() call if batch size increases (and thus requires
        // larger workspace).
        const INIT_BATCH_SIZE: i32 = 1;

        let cudnn_framework = self.framework().cudnn();
        let input_mode = input_mode.as_cudnn()?;
        let network_mode = network_mode.as_cudnn()?;
        let algorithm = algorithm.as_cudnn()?;

        let data_type = <T as DataTypeInfo>::cudnn_data_type();

        let drop_desc = exec2!(cudnn_framework.init_dropout(
            dropout_probability.unwrap_or(0.5),
            dropout_seed.unwrap_or(0),
        ) => "Unable to create Dropout Layer")?;

        let dropout_memory: cudnnDropoutDescriptor_t = *drop_desc.dropout_desc().id_c();

        let x_desc = rnn_sequence_descriptors(
            sequence_length,
            input_size,
            hidden_size,
            INIT_BATCH_SIZE,
            num_layers,
            direction_mode,
            data_type,
        )?
        .x_desc;

        let direction_mode = direction_mode.as_cudnn()?;

        let rnn_desc = exec2!(RnnDescriptor::new(
            &cudnn_framework,
            hidden_size,
            num_layers,
            drop_desc,
            input_mode,
            direction_mode,
            network_mode,
            algorithm,
            data_type,
            (RnnPaddingMode::Disabled).as_cudnn().unwrap(),
        ) => "Failed to create RNN descriptor")?;

        exec2!(cudnn_framework.init_rnn(
            &x_desc,
            rnn_desc,
            hidden_size,
            num_layers,
            sequence_length,
            dropout_memory,
            input_mode,
            direction_mode,
            network_mode,
            algorithm,
            <T as DataTypeInfo>::cudnn_data_type(),
            MathType::TensorOPMathAllowConversion.as_cudnn()?,
        ) => "Unable to perform RNN Initialization")
    }

    /// Train and Output a RNN Network
    fn rnn_forward(
        &self,
        src: &SharedTensor<T>,
        output: &mut SharedTensor<T>,
        rnn_config: &mut Self::CRNN,
        weight: &SharedTensor<T>,
    ) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();

        log::trace!("rnn_forward: src[dims] = {:?}", src.desc());
        log::trace!("rnn_forward: output[dims] = {:?}", output.desc());
        log::trace!("rnn_forward: weight[dims] = {:?}", weight.desc());

        let src_dimensions = src.desc();
        let sequence_descriptors = rnn_sequence_descriptors(
            rnn_config.sequence_length,
            src_dimensions[1] as i32,
            rnn_config.hidden_size,
            src_dimensions[0] as i32,
            rnn_config.num_layers,
            DirectionMode::UniDirectional, // FIXME make it configurable
            <T as DataTypeInfo>::cudnn_data_type(),
        )?;

        let weight_desc = weight.cudnn_filter_desc()?;

        // Resize workspace if necessary.
        exec2!(cudnn_framework.maybe_resize_rnn(rnn_config, &sequence_descriptors.x_desc) => "Unable to resize RNN buffers")?;

        let src_mem = read!(src, self);
        let weight_mem = weight.read(self.device()).unwrap();
        let output_mem = output.write_only(self.device()).unwrap();

        exec2!(cudnn_framework.rnn_forward::<f32>(
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
            *rnn_config.workspace.id_c(),
            *rnn_config.training_reserve.id_c(),
        )  => "Unable to perform RNN Forward")
    }

    fn rnn_backward_data(
        &self,
        src: &SharedTensor<T>,
        src_gradient: &mut SharedTensor<T>,
        output: &SharedTensor<T>,
        output_gradient: &SharedTensor<T>,
        rnn_config: &Self::CRNN,
        weight: &SharedTensor<T>,
    ) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let src_dimensions = src.desc().clone();
        let sequence_descriptors = rnn_sequence_descriptors(
            rnn_config.sequence_length,
            src_dimensions[1] as i32,
            rnn_config.hidden_size,
            src_dimensions[0] as i32,
            rnn_config.num_layers,
            DirectionMode::UniDirectional,
            <T as DataTypeInfo>::cudnn_data_type(),
        )?;
        let weight_desc = weight.cudnn_filter_desc()?;

        let _src_mem = read!(src, self);
        let src_gradient_mem = write_only!(src_gradient, self);
        let weight_mem = read!(weight, self);
        let output_mem = read!(output, self);
        let output_gradient_mem = read!(output_gradient, self);
        exec2!(cudnn_framework.rnn_backward_data::<f32>(
            rnn_config,
            sequence_descriptors.y_desc,
            trans!(output_mem),
            sequence_descriptors.dy_desc,
            //output_gradient,
            trans!(output_gradient_mem),
            &sequence_descriptors.dhy_desc,
            //final_hidden_gradient,
            std::ptr::null_mut(),
            &sequence_descriptors.dcy_desc,
            //final_cell_gradient,
            std::ptr::null_mut(),
            &weight_desc,
            trans!(weight_mem),
            &sequence_descriptors.hx_desc,
            std::ptr::null(),
            &sequence_descriptors.cx_desc,
            std::ptr::null(),
            sequence_descriptors.dx_desc,
            trans_mut!(src_gradient_mem),
            &sequence_descriptors.dhx_desc,
            std::ptr::null_mut(),
            &sequence_descriptors.dcx_desc,
            std::ptr::null_mut(),
            *rnn_config.workspace.id_c(),
            *rnn_config.training_reserve.id_c(),
        ) => "Unable to execute CUDA cuDNN RNN Backward Data")
    }

    fn rnn_backward_weights(
        &self,
        src: &SharedTensor<T>,
        output: &SharedTensor<T>,
        filter: &mut SharedTensor<T>,
        rnn_config: &Self::CRNN,
    ) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let src_dimensions = src.desc().clone();
        let sequence_descriptors = rnn_sequence_descriptors(
            rnn_config.sequence_length,
            src_dimensions[1] as i32,
            rnn_config.hidden_size,
            src_dimensions[0] as i32,
            rnn_config.num_layers,
            DirectionMode::UniDirectional,
            <T as DataTypeInfo>::cudnn_data_type(),
        )?;
        let filter_desc = filter.cudnn_filter_desc()?;
        let src_mem = read!(src, self);
        let output_mem = read!(output, self);
        let filter_mem = write_only!(filter, self);
        exec2!(cudnn_framework.rnn_backward_weights::<f32>(
            rnn_config,
            sequence_descriptors.x_desc,
            trans!(src_mem),
            &sequence_descriptors.hx_desc,
            std::ptr::null_mut(),
            sequence_descriptors.y_desc,
            trans!(output_mem),
            filter_desc,
            trans_mut!(filter_mem),
            *rnn_config.workspace.id_c(),
            *rnn_config.training_reserve.id_c(),
        )  => "Unable to execute CUDA cuDNN RNN Backward Data")
    }
}

impl<T> SigmoidPointwise<T> for Backend<Cuda>
where
    T: Float + Default + DataTypeInfo,
{
    fn sigmoid_pointwise(&self, x: &mut SharedTensor<T>) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let x_desc = x.cudnn_tensor_desc_flat()?;
        let x_mem = read_write!(x, self);

        exec2!(cudnn_framework.sigmoid_forward(
            &cudnn_framework.init_activation().unwrap(),
            &x_desc,
            trans!(x_mem),
            &x_desc,
            trans_mut!(x_mem),
            scal_params,
        )  => "Unable to execute CUDA cuDNN Sigmoid Pointwise forward.")
    }

    fn sigmoid_pointwise_grad(
        &self,
        x: &SharedTensor<T>,
        x_diff: &mut SharedTensor<T>,
    ) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let x_desc = x.cudnn_tensor_desc_flat()?;
        let dx_desc = x_diff.cudnn_tensor_desc_flat()?;
        let x_mem = read!(x, self);
        let dx_mem = read_write!(x_diff, self);
        // TODO move config one level up
        exec2!(cudnn_framework.sigmoid_backward(
            &cudnn_framework.init_activation().unwrap(),
            &x_desc,
            trans!(x_mem),
            &dx_desc,
            trans!(dx_mem),
            &x_desc,
            trans!(x_mem),
            &dx_desc,
            trans_mut!(dx_mem),
            scal_params,
        )  => "Unable to execute CUDA cuDNN Sigmoid Pointwise backward.")
    }
}

impl<T> Relu<T> for Backend<Cuda>
where
    T: Float + Default + DataTypeInfo,
{
    fn relu(&self, x: &SharedTensor<T>, result: &mut SharedTensor<T>) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let r_desc = result.cudnn_tensor_desc_flat()?;
        let x_mem = read!(x, self);
        let r_mem = write_only!(result, self);
        exec2!(cudnn_framework.relu_forward(
            &cudnn_framework.init_activation().unwrap(),
            &x.cudnn_tensor_desc_flat()?,
            trans!(x_mem),
            &r_desc,
            trans_mut!(r_mem),
            scal_params,
        )  => "Unable to execute CUDA cuDNN Activation relu Forward.")
    }

    fn relu_grad(
        &self,
        x: &SharedTensor<T>,
        x_diff: &SharedTensor<T>,
        result: &SharedTensor<T>,
        result_diff: &mut SharedTensor<T>,
    ) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let dr_desc = result_diff.cudnn_tensor_desc_flat()?;
        let x_mem = read!(x, self);
        let dx_mem = read!(x_diff, self);
        let r_mem = read!(result, self);
        let dr_mem = write_only!(result_diff, self);

        exec2!(cudnn_framework.relu_backward(
            &cudnn_framework.init_activation().unwrap(),
            &x.cudnn_tensor_desc_flat()?,
            trans!(x_mem),
            &x_diff.cudnn_tensor_desc_flat()?,
            trans!(dx_mem),
            &result.cudnn_tensor_desc_flat()?,
            trans!(r_mem),
            &dr_desc,
            trans_mut!(dr_mem),
            scal_params,
        )  => "Unable to execute CUDA cuDNN Activation relu Backward.")
    }
}

impl<T> ReluPointwise<T> for Backend<Cuda>
where
    T: Float + Default + DataTypeInfo,
{
    fn relu_pointwise(&self, x: &mut SharedTensor<T>) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let x_desc = x.cudnn_tensor_desc_flat()?;
        let x_mem = read_write!(x, self);

        exec2!(cudnn_framework.relu_forward(
            &cudnn_framework.init_activation().unwrap(),
            &x_desc,
            trans!(x_mem),
            &x_desc,
            trans_mut!(x_mem),
            scal_params,
        )  => "Unable to execute CUDA cuDNN ReLU Pointwise forward.")
    }

    fn relu_pointwise_grad(
        &self,
        x: &SharedTensor<T>,
        x_diff: &mut SharedTensor<T>,
    ) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let x_desc = x.cudnn_tensor_desc_flat()?;
        let dx_desc = x_diff.cudnn_tensor_desc_flat()?;
        let x_mem = read!(x, self);
        let dx_mem = read_write!(x_diff, self);

        exec2!(cudnn_framework.relu_backward(
            &cudnn_framework.init_activation().unwrap(),
            &x_desc,
            trans!(x_mem),
            &dx_desc,
            trans!(dx_mem),
            &x_desc,
            trans!(x_mem),
            &dx_desc,
            trans_mut!(dx_mem),
            scal_params,
        )  => "Unable to execute CUDA cuDNN ReLU Pointwise backward.")
    }
}

impl<T> Tanh<T> for Backend<Cuda>
where
    T: Float + Default + DataTypeInfo,
{
    fn tanh(&self, x: &SharedTensor<T>, result: &mut SharedTensor<T>) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let r_desc = result.cudnn_tensor_desc_flat()?;
        let x_mem = read!(x, self);
        let r_mem = write_only!(result, self);
        exec2!(cudnn_framework.tanh_forward(
            &cudnn_framework.init_activation().unwrap(),
            &x.cudnn_tensor_desc_flat()?,
            trans!(x_mem),
            &r_desc,
            trans_mut!(r_mem),
            scal_params,
        )  => "Unable to execute CUDA cuDNN Activation tanh Forward.")
    }

    fn tanh_grad(
        &self,
        x: &SharedTensor<T>,
        x_diff: &SharedTensor<T>,
        result: &SharedTensor<T>,
        result_diff: &mut SharedTensor<T>,
    ) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let dr_desc = result_diff.cudnn_tensor_desc_flat()?;
        let x_mem = read!(x, self);
        let dx_mem = read!(x_diff, self);
        let r_mem = read!(result, self);
        let dr_mem = write_only!(result_diff, self);
        exec2!(cudnn_framework.tanh_backward(
            &cudnn_framework.init_activation().unwrap(),
            &x.cudnn_tensor_desc_flat()?,
            trans!(x_mem),
            &x_diff.cudnn_tensor_desc_flat()?,
            trans!(dx_mem),
            &result.cudnn_tensor_desc_flat()?,
            trans!(r_mem),
            &dr_desc,
            trans_mut!(dr_mem),
            scal_params,
        )  => "Unable to execute CUDA cuDNN Activation tanh Backward.")
    }
}

impl<T> TanhPointwise<T> for Backend<Cuda>
where
    T: Float + Default + DataTypeInfo,
{
    fn tanh_pointwise(&self, x: &mut SharedTensor<T>) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let x_desc = x.cudnn_tensor_desc_flat()?;
        let x_mem = read_write!(x, self);
        exec2!(cudnn_framework.tanh_forward(
            &cudnn_framework.init_activation().unwrap(),
            &x_desc,
            trans!(x_mem),
            &x_desc,
            trans_mut!(x_mem),
            scal_params,
        )  => "Unable to execute CUDA cuDNN Tanh Pointwise forward.")
    }

    fn tanh_pointwise_grad(
        &self,
        x: &SharedTensor<T>,
        x_diff: &mut SharedTensor<T>,
    ) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let x_desc = x.cudnn_tensor_desc_flat()?;
        let dx_desc = x_diff.cudnn_tensor_desc_flat()?;
        let x_mem = read!(x, self);
        let dx_mem = read_write!(x_diff, self);
        exec2!(cudnn_framework.tanh_backward(
            &cudnn_framework.init_activation().unwrap(),
            &x_desc,
            trans!(x_mem),
            &dx_desc,
            trans!(dx_mem),
            &x_desc,
            trans!(x_mem),
            &dx_desc,
            trans_mut!(dx_mem),
            scal_params,
        )  => "Unable to execute CUDA cuDNN Tanh Pointwise backward.")
    }
}

impl<T> Softmax<T> for Backend<Cuda>
where
    T: Float + Default + DataTypeInfo,
{
    fn softmax(&self, x: &SharedTensor<T>, result: &mut SharedTensor<T>) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let r_desc = result.cudnn_tensor_desc_softmax()?;
        let x_mem = read!(x, self);
        let r_mem = write_only!(result, self);
        exec2!(cudnn_framework.softmax_forward(
            &x.cudnn_tensor_desc_softmax()?,
            trans!(x_mem),
            &r_desc,
            trans_mut!(r_mem),
            scal_params,
        )  => "Unable to execute CUDA cuDNN softmax Forward.")
    }

    fn softmax_grad(
        &self,
        x: &SharedTensor<T>,
        x_diff: &SharedTensor<T>,
        result_diff: &mut SharedTensor<T>,
    ) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let dr_desc = result_diff.cudnn_tensor_desc_softmax()?;
        let x_mem = read!(x, self);
        let dx_mem = read!(x_diff, self);
        let dr_mem = write_only!(result_diff, self);
        exec2!(cudnn_framework.softmax_backward(
            &x.cudnn_tensor_desc_softmax()?,
            trans!(x_mem),
            &x_diff.cudnn_tensor_desc_softmax()?,
            trans!(dx_mem),
            &dr_desc,
            trans_mut!(dr_mem),
            scal_params,
        )  => "Unable to execute CUDA cuDNN softmax Backward.")
    }
}

impl<T> LogSoftmax<T> for Backend<Cuda>
where
    T: Float + Default + DataTypeInfo,
{
    fn log_softmax(&self, x: &SharedTensor<T>, result: &mut SharedTensor<T>) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let r_desc = result.cudnn_tensor_desc_softmax()?;
        let x_mem = read!(x, self);
        let r_mem = write_only!(result, self);
        exec2!(cudnn_framework.log_softmax_forward(
            &x.cudnn_tensor_desc_softmax()?,
            trans!(x_mem),
            &r_desc,
            trans_mut!(r_mem),
            scal_params,
        )  => "Unable to execute CUDA cuDNN softmax Forward.")
    }
    fn log_softmax_grad(
        &self,
        x: &SharedTensor<T>,
        x_diff: &SharedTensor<T>,
        result_diff: &mut SharedTensor<T>,
    ) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let dr_desc = result_diff.cudnn_tensor_desc_softmax()?;
        let x_mem = read!(x, self);
        let dx_mem = read!(x_diff, self);
        let dr_mem = write_only!(result_diff, self);
        exec2!(cudnn_framework.log_softmax_backward(
            &x.cudnn_tensor_desc_softmax()?,
            trans!(x_mem),
            &x_diff.cudnn_tensor_desc_softmax()?,
            trans!(dx_mem),
            &dr_desc,
            trans_mut!(dr_mem),
            scal_params,
        ) => "Unable to execute CUDA cuDNN log softmax Backward.")
    }
}

impl<T> LRN<T> for Backend<Cuda>
where
    T: Float + Default + DataTypeInfo,
{
    fn new_lrn_config(&self, n: u32, alpha: f64, beta: f64, k: f64) -> Result<Self::CLRN, Error> {
        let cudnn_framework = self.framework().cudnn();
        Ok(cudnn_framework
            .init_normalization(n, alpha, beta, k)
            .unwrap())
    }

    fn lrn(
        &self,
        x: &SharedTensor<T>,
        result: &mut SharedTensor<T>,
        config: &Self::CLRN,
    ) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let r_desc = result.cudnn_tensor_desc()?;
        let x_mem = read!(x, self);
        let r_mem = write_only!(result, self);
        exec2!(cudnn_framework.lrn_forward(
                config,
                &x.cudnn_tensor_desc()?,
                trans!(x_mem),
                &r_desc,
                trans_mut!(r_mem),
                scal_params,
            ) => "Unable to execute CUDA cuDNN Activation lrn Forward."
        )
    }

    #[allow(unused_variables)]
    fn lrn_grad(
        &self,
        x: &SharedTensor<T>,
        x_diff: &SharedTensor<T>,
        result: &SharedTensor<T>,
        result_diff: &mut SharedTensor<T>,
        config: &Self::CLRN,
    ) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let dr_desc = result_diff.cudnn_tensor_desc()?;
        let x_mem = read!(x, self);
        let dx_mem = read!(x_diff, self);
        let r_mem = read!(result, self);
        let dr_mem = write_only!(result_diff, self);
        exec2!(cudnn_framework.lrn_backward(
            config,
            &x.cudnn_tensor_desc()?,
            trans!(x_mem),
            &x_diff.cudnn_tensor_desc()?,
            trans!(dx_mem),
            &result.cudnn_tensor_desc()?,
            trans!(r_mem),
            &dr_desc,
            trans_mut!(dr_mem),
            scal_params,
        ) => "Unable to execute CUDA cuDNN Activation lrn Backward.")
    }
}

impl<T> Pooling<T> for Backend<Cuda>
where
    T: Float + Default + DataTypeInfo,
{
    fn new_pooling_config(
        &self,
        window: &[i32],
        stride: &[i32],
        padding: &[i32],
    ) -> Result<Self::CPOOL, Error> {
        let pooling_avg = crate::cudnn::PoolingDescriptor::new(
            crate::cudnn::cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
            window,
            padding,
            stride,
        )
        .unwrap();
        let pooling_max = crate::cudnn::PoolingDescriptor::new(
            crate::cudnn::cudnnPoolingMode_t::CUDNN_POOLING_MAX,
            window,
            padding,
            stride,
        )
        .unwrap();
        Ok(crate::cudnn::utils::PoolingConfig::new(
            pooling_avg,
            pooling_max,
        ))
    }

    fn pooling_max(
        &self,
        x: &SharedTensor<T>,
        result: &mut SharedTensor<T>,
        config: &Self::CPOOL,
    ) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();

        let r_desc = result.cudnn_tensor_desc()?;
        let x_mem = read!(x, self);
        let r_mem = write_only!(result, self);
        exec2!(cudnn_framework.pooling_max_forward(
                config,
                &x.cudnn_tensor_desc()?,
                trans!(x_mem),
                &r_desc,
                trans_mut!(r_mem),
                scal_params,
            ) => "Unable to execute CUDA cuDNN max pooling Forward."
        )
    }

    #[allow(unused_variables)]
    fn pooling_max_grad(
        &self,
        x: &SharedTensor<T>,
        x_diff: &SharedTensor<T>,
        result: &SharedTensor<T>,
        result_diff: &mut SharedTensor<T>,
        config: &Self::CPOOL,
    ) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let dr_desc = result_diff.cudnn_tensor_desc()?;
        let x_mem = read!(x, self);
        let dx_mem = read!(x_diff, self);
        let r_mem = read!(result, self);
        let dr_mem = write_only!(result_diff, self);
        exec2!(cudnn_framework.pooling_max_backward(
            config,
            &x.cudnn_tensor_desc()?,
            trans!(x_mem),
            &x_diff.cudnn_tensor_desc()?,
            trans!(dx_mem),
            &result.cudnn_tensor_desc()?,
            trans!(r_mem),
            &dr_desc,
            trans_mut!(dr_mem),
            scal_params,
        )  => "Unable to execute CUDA cuDNN max pooling Backward.")
    }

    fn pooling_avg(
        &self,
        x: &SharedTensor<T>,
        result: &mut SharedTensor<T>,
        config: &Self::CPOOL,
    ) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let r_desc = result.cudnn_tensor_desc()?;
        let x_mem = read!(x, self);
        let r_mem = write_only!(result, self);
        exec2!(cudnn_framework.pooling_avg_forward(
            config,
            &x.cudnn_tensor_desc()?,
            trans!(x_mem),
            &r_desc,
            trans_mut!(r_mem),
            scal_params,
        )  => "Unable to execute CUDA cuDNN avg pooling Forward.")
    }

    #[allow(unused_variables)]
    fn pooling_avg_grad(
        &self,
        x: &SharedTensor<T>,
        x_diff: &SharedTensor<T>,
        result: &SharedTensor<T>,
        result_diff: &mut SharedTensor<T>,
        config: &Self::CPOOL,
    ) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let scal_params: crate::cudnn::utils::ScalParams<T> =
            crate::cudnn::utils::ScalParams::default();
        let dr_desc = result_diff.cudnn_tensor_desc()?;
        let x_mem = read!(x, self);
        let dx_mem = read!(x_diff, self);
        let r_mem = read!(result, self);
        let dr_mem = write_only!(result_diff, self);
        exec2!(cudnn_framework.pooling_avg_backward(
            config,
            &x.cudnn_tensor_desc()?,
            trans!(x_mem),
            &x_diff.cudnn_tensor_desc()?,
            trans!(dx_mem),
            &result.cudnn_tensor_desc()?,
            trans!(r_mem),
            &dr_desc,
            trans_mut!(dr_mem),
            scal_params,
        ) => "Unable to execute CUDA cuDNN avg pooling Backward.")
    }
}

impl<T> Dropout<T> for Backend<Cuda>
where
    T: Float + Default + DataTypeInfo,
{
    fn new_dropout_config(&self, probability: f32, seed: u64) -> Result<Self::CDROP, Error> {
        let cudnn_framework = self.framework().cudnn();
        Ok(cudnn_framework.init_dropout(probability, seed).unwrap())
    }

    fn dropout(
        &self,
        x: &SharedTensor<T>,
        result: &mut SharedTensor<T>,
        config: &Self::CDROP,
    ) -> Result<(), Error> {
        let cudnn_framework = self.framework().cudnn();
        let r_desc = result.cudnn_tensor_desc()?;
        let x_mem = read!(x, self);
        let r_mem = write_only!(result, self);

        exec2!(cudnn_framework.dropout_forward::<f32>(
            config,
            &x.cudnn_tensor_desc()?,
            trans!(x_mem),
            &r_desc,
            trans_mut!(r_mem),
        ) => "Unable to execute CUDA cuDNN Dropout Forward.")
    }

    #[allow(unused_variables)]
    fn dropout_grad(
        &self,
        x: &SharedTensor<T>,
        x_diff: &SharedTensor<T>,
        result: &SharedTensor<T>,
        result_diff: &mut SharedTensor<T>,
        config: &Self::CDROP,
    ) -> Result<(), Error> {
        // TODO what to do with the gradient? should be all zeroes since this is supposed to be a `nop` but I am not 100% sure about the nv implementations
        // let dr_desc = result_diff.cudnn_tensor_desc()?;
        // let x_mem = read!(x, self);
        // let dx_mem = read!(x_diff, self);
        // let r_mem = write_only!(result, self);
        // let dr_mem = write_only!(result_diff, self);
        // exec2!(cudnn_framework.dropout_backward::<f32>(config,
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

#[derive(Debug, thiserror::Error)]
pub enum WrappingError {
    #[error("{0}")]
    Misc(&'static str),

    #[error(transparent)]
    Inner(#[from] rcudnn::Error),
}

impl Into<PluginError> for WrappingError {
    fn into(self) -> PluginError {
        PluginError::PluginInner(Box::new(self))
    }
}

impl Into<co::Error> for WrappingError {
    fn into(self) -> co::Error {
        co::Error::Plugin(co::plugin::Error::PluginInner(self.into()))
    }
}
