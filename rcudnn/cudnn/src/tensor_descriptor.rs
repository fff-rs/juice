//! Defines a Tensor Descriptor.
//!
//! A Tensor Descriptor is used to hold information about the data,
//! which is needed for the operations to obtain information about
//! the structure and dimensionality of the data.

use super::utils::DataType;
use super::{Error, API};
use crate::ffi::*;

#[derive(Debug, Clone)]
/// Describes a TensorDescriptor.
pub struct TensorDescriptor {
    id: cudnnTensorDescriptor_t,
}

/// Return C Handle for a Vector of Tensor Descriptors
pub fn tensor_vec_id_c(tensor_vec: &[TensorDescriptor]) -> Vec<cudnnTensorDescriptor_t> {
    tensor_vec.iter().map(|tensor| *tensor.id_c()).collect()
}

impl Drop for TensorDescriptor {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        API::destroy_tensor_descriptor(*self.id_c());
    }
}

impl TensorDescriptor {
    /// Initializes a new CUDA cuDNN Tensor Descriptor.
    pub fn new(
        dims: &[i32],
        strides: &[i32],
        data_type: DataType,
    ) -> Result<TensorDescriptor, Error> {
        // CUDA supports tensors with 3 to 8 dimensions. If the actual tensor has less dimensions,
        // CUDA recommends setting several first dimensions to 1.
        const MIN_DIMS: i32 = 3;

        let mut cuda_dims = [0; 8];
        let mut cuda_strides = [0; 8];
        let mut cuda_dim_count = dims.len() as i32;
        if cuda_dim_count < MIN_DIMS {
            let stub_dim_count = MIN_DIMS - cuda_dim_count;
            cuda_dim_count = MIN_DIMS;
            for i in 0..stub_dim_count {
                cuda_dims[i as usize] = 1;
            }
            for i in 0..dims.len() {
                cuda_dims[i + stub_dim_count as usize] = dims[i];
                cuda_strides[i + stub_dim_count as usize] = strides[i];
            }
        } else {
            for i in 0..dims.len() {
                cuda_dims[i] = dims[i];
                cuda_strides[i] = strides[i];
            }
        }

        let dims_ptr = cuda_dims.as_ptr();
        let strides_ptr = cuda_strides.as_ptr();
        let generic_tensor_desc = API::create_tensor_descriptor()?;
        let data_type = API::cudnn_data_type(data_type);

        // assert!(false, "{:?}, {}", cuda_dims, cuda_dim_count);

        API::set_tensor_descriptor(
            generic_tensor_desc,
            data_type,
            cuda_dim_count,
            dims_ptr,
            strides_ptr,
        )?;
        Ok(TensorDescriptor::from_c(generic_tensor_desc))
    }

    /// Initializes a new CUDA cuDNN Tensor Descriptor from its C type.
    pub fn from_c(id: cudnnTensorDescriptor_t) -> TensorDescriptor {
        TensorDescriptor { id }
    }

    /// Returns the CUDA cuDNN Tensor Descriptor as its C type.
    pub fn id_c(&self) -> &cudnnTensorDescriptor_t {
        &self.id
    }
}
