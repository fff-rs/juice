//! Defines a Convolution Descriptor.
//!
//! A Convolution Descriptor is used to hold information about the convolution,
//! which is needed for forward and backward convolutional operations.

use super::utils::DataType;
use super::{Error, API};
use ffi::*;

#[derive(Debug, Clone)]
/// Describes a Convolution Descriptor.
pub struct ConvolutionDescriptor {
    id: cudnnConvolutionDescriptor_t,
}

impl Drop for ConvolutionDescriptor {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        API::destroy_convolution_descriptor(*self.id_c());
    }
}

impl ConvolutionDescriptor {
    /// Initializes a new CUDA cuDNN ConvolutionDescriptor.
    pub fn new(
        pad: &[i32],
        filter_stride: &[i32],
        data_type: DataType,
    ) -> Result<ConvolutionDescriptor, Error> {
        let array_length = pad.len() as i32;
        let upscale: Vec<i32> = ::std::iter::repeat(1i32)
            .take(array_length as usize)
            .collect();

        let generic_convolution_desc = API::create_convolution_descriptor()?;
        match data_type {
            DataType::Float => {
                let d_type = cudnnDataType_t::CUDNN_DATA_FLOAT;
                API::set_convolution_descriptor(
                    generic_convolution_desc,
                    d_type,
                    cudnnConvolutionMode_t::CUDNN_CONVOLUTION,
                    array_length,
                    pad.as_ptr(),
                    filter_stride.as_ptr(),
                    upscale.as_ptr(),
                )?;
                Ok(ConvolutionDescriptor::from_c(generic_convolution_desc))
            }
            DataType::Double => {
                let d_type = cudnnDataType_t::CUDNN_DATA_DOUBLE;
                API::set_convolution_descriptor(
                    generic_convolution_desc,
                    d_type,
                    cudnnConvolutionMode_t::CUDNN_CONVOLUTION,
                    array_length,
                    pad.as_ptr(),
                    filter_stride.as_ptr(),
                    upscale.as_ptr(),
                )?;
                Ok(ConvolutionDescriptor::from_c(generic_convolution_desc))
            }
            DataType::Half => {
                let d_type = cudnnDataType_t::CUDNN_DATA_FLOAT;
                API::set_convolution_descriptor(
                    generic_convolution_desc,
                    d_type,
                    cudnnConvolutionMode_t::CUDNN_CONVOLUTION,
                    array_length,
                    pad.as_ptr(),
                    filter_stride.as_ptr(),
                    upscale.as_ptr(),
                )?;
                Ok(ConvolutionDescriptor::from_c(generic_convolution_desc))
            }
        }
    }

    /// Initializes a new CUDA cuDNN ConvolutionDescriptor from its C type.
    pub fn from_c(id: cudnnConvolutionDescriptor_t) -> ConvolutionDescriptor {
        ConvolutionDescriptor { id: id }
    }

    /// Returns the CUDA cuDNN ConvolutionDescriptor as its C type.
    pub fn id_c(&self) -> &cudnnConvolutionDescriptor_t {
        &self.id
    }
}
