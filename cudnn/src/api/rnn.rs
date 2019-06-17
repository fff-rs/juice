//! Provides the RNN functionality from the CUDA cuDNN API.
//!
//! Includes the RNN functionality.

use crate::ffi::*;
use crate::{Error, API};

impl API {
    //  ///
    //  /// cuDNN RNN Configuration
    //  ///

    /// Returns the workspace size in byte, which are needed for the given rnnal algorithm.
    pub fn get_rnn_workspace_size(
        handle: cudnnHandle_t,
        rnn_desc: cudnnRNNDescriptor_t,
        unroll_sequence_length: i32,
        x_desc: Vec<cudnnTensorDescriptor_t>,
    ) -> Result<usize, Error> {
        unsafe {
            API::ffi_get_rnn_workspace_size(
                handle,
                rnn_desc,
                unroll_sequence_length,
                x_desc.as_slice(),
            )
        }
    }

    unsafe fn ffi_get_rnn_workspace_size(
        handle: cudnnHandle_t,
        rnn_desc: cudnnRNNDescriptor_t,
        unroll_sequence_length: i32,
        x_desc: &[cudnnTensorDescriptor_t],
    ) -> Result<::libc::size_t, Error> {
        let mut size: ::libc::size_t = 0;
        let size_ptr: *mut ::libc::size_t = &mut size;
        match cudnnGetRNNWorkspaceSize(handle, rnn_desc, unroll_sequence_length, x_desc.as_ptr(), size_ptr) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(size),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("At least one of the following conditions are met: One of the parameters `handle`, `x_desc`, `rnn_desc` is NULL. The tensors in `x_desc` are not of the same data type. The batch size of the tensors `x_desc` are not decreasing or staying constant.")),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("The data type used in `src_desc` is not supported for RNN.")),
            _ => Err(Error::Unknown("Unable to get CUDA cuDNN RNN Forward Workspace size.")),
        }
    }

    //     cudnnStatus_t
    // cudnnGetRNNParamsSize( cudnnHandle_t
    // const cudnnRNNDescriptor_t
    // const cudnnTensorDescriptor_t
    // size_t
    // cudnnDataType_t dataType)

    // cudnnStatus_t
    // cudnnGetRNNTrainingReserveSize( cudnnHandle_t
    // const cudnnRNNDescriptor_t
    // const int seqLength,
    // const cudnnTensorDescriptor_t
    // size_t
    // handle,
    // rnnDesc,
    // *xDesc,
    // *sizeInBytes)

    //
    // cuDNN RNN
    //

    /// Creates a generic CUDA cuDNN RNN Descriptor.
    pub fn create_rnn_descriptor() -> Result<cudnnRNNDescriptor_t, Error> {
        unsafe { API::ffi_create_rnn_descriptor() }
    }

    /// Destroys a CUDA cuDNN RNN Descriptor.
    ///
    /// Should be called when freeing a CUDA::Descriptor to not trash up the CUDA device.
    pub fn destroy_rnn_descriptor(desc: cudnnRNNDescriptor_t) -> Result<(), Error> {
        unsafe { API::ffi_destroy_rnn_descriptor(desc) }
    }

    /// Initializes a generic CUDA cuDNN RNN Descriptor with specific properties.
    pub fn set_rnn_descriptor(
        handle: cudnnHandle_t,
        desc: cudnnRNNDescriptor_t,
        hidden_size: i32,
        num_layers: i32,
        dropout_desc: cudnnDropoutDescriptor_t,
        input_mode: cudnnRNNInputMode_t,
        direction: cudnnDirectionMode_t,
        mode: cudnnRNNMode_t,
        algorithm: cudnnRNNAlgo_t,
        data_type: cudnnDataType_t,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_set_rnn_descriptor(
                handle,
                desc,
                hidden_size,
                num_layers,
                dropout_desc,
                input_mode,
                direction,
                mode,
                algorithm,
                data_type,
            )
        }
    }

    unsafe fn ffi_set_rnn_descriptor(
        handle: cudnnHandle_t,
        desc: cudnnRNNDescriptor_t,
        hidden_size: i32,
        num_layers: i32,
        dropout_desc: cudnnDropoutDescriptor_t,
        input_mode: cudnnRNNInputMode_t,
        direction: cudnnDirectionMode_t,
        mode: cudnnRNNMode_t,
        algorithm: cudnnRNNAlgo_t,
        data_type: cudnnDataType_t,
    ) -> Result<(), Error> {
        match cudnnSetRNNDescriptor(
            handle,
            desc,
            hidden_size,
            num_layers,
            dropout_desc,
            input_mode,
            direction,
            mode,
            algorithm,
            data_type,
        ) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("FIXME RNN")),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("FIXME RNN")),
            _ => Err(Error::Unknown("Unable to set CUDA cuDNN RNN Descriptor.")),
        }
    }

    unsafe fn ffi_create_rnn_descriptor() -> Result<cudnnRNNDescriptor_t, Error> {
        let mut rnn_desc: cudnnRNNDescriptor_t = ::std::ptr::null_mut();
        match cudnnCreateRNNDescriptor(&mut rnn_desc) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(rnn_desc),
            cudnnStatus_t::CUDNN_STATUS_ALLOC_FAILED => {
                Err(Error::AllocFailed("The resources could not be allocated"))
            }
            _ => Err(Error::Unknown(
                "Unable create generic CUDA cuDNN RNN Descriptor",
            )),
        }
    }

    unsafe fn ffi_destroy_rnn_descriptor(rnn_desc: cudnnRNNDescriptor_t) -> Result<(), Error> {
        match cudnnDestroyRNNDescriptor(rnn_desc) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            _ => Err(Error::Unknown(
                "Unable to destroy CUDA cuDNN Dropout Descriptor",
            )),
        }
    }
}

// cudnnStatus_t
// cudnnRNNForwardInference( cudnnHandle_t handle,
// const cudnnRNNDescriptor_t rnnDesc,
// const int seqLength,
// const cudnnTensorDescriptor_t * xDesc,
// const void * x,
// const cudnnTensorDescriptor_t hxDesc,
// const void * hx,
// const cudnnTensorDescriptor_t cxDesc,
// const void * cx,
// const cudnnFilterDescriptor_t wDesc,
// const void * w,
// const cudnnTensorDescriptor_t *yDesc,
// void * y,
// const cudnnTensorDescriptor_t hyDesc,
// void * hy,
// const cudnnTensorDescriptor_t cyDesc,
// void * cy,
// void * workspace,
// size_t workSpaceSizeInBytes)

// cudnnStatus_t
// cudnnRNNForwardTraining( cudnnHandle_t handle,
// const cudnnRNNDescriptor_t rnnDesc,
// const int seqLength,
// const cudnnTensorDescriptor_t *xDesc,
// const void * x,
// const cudnnTensorDescriptor_t hxDesc,
// const void * hx,
// const cudnnTensorDescriptor_t cxDesc,
// const void * cx,
// const cudnnFilterDescriptor_t wDesc,
// const void * w,
// const cudnnTensorDescriptor_t *yDesc,
// void * y,
// const cudnnTensorDescriptor_t hyDesc,
// void * hy,
// const cudnnTensorDescriptor_t cyDesc,
// void * cy,
// void * workspace,
// size_t workSpaceSizeInBytes,
// void * reserveSpace,
// size_t reserveSpaceSizeInBytes)

//  /// Computes a rnn forward function.
//     pub fn rnn_forward(
//         handle: cudnnHandle_t,
//         algo: cudnnRNNFwdAlgo_t,
//         conv_desc: cudnnRNNDescriptor_t,
//         work_space: *mut ::libc::c_void,
//         work_size_in_bytes: ::libc::size_t,
//         alpha: *const ::libc::c_void,
//         src_desc: cudnnTensorDescriptor_t,
//         src_data: *const ::libc::c_void,
//         rnn_desc: cudnnRNNDescriptor_t,
//         rnn_data: *const ::libc::c_void,
//         beta: *const ::libc::c_void,
//         dest_desc: cudnnTensorDescriptor_t,
//         dest_data: *mut ::libc::c_void,
//     ) -> Result<(), Error> {
//         unsafe { API::ffi_rnn_forward(handle, alpha, src_desc, src_data, rnn_desc, rnn_data, conv_desc, algo, work_space, work_size_in_bytes, beta, dest_desc, dest_data) }
//     }

//  /// Computes a rnn backward function w.r.t the bias.
//     pub fn rnn_backward_bias(
//         handle: cudnnHandle_t,
//         alpha: *const ::libc::c_void,
//         src_desc: cudnnTensorDescriptor_t,
//         src_data: *const ::libc::c_void,
//         beta: *const ::libc::c_void,
//         dest_desc: cudnnTensorDescriptor_t,
//         dest_data: *mut ::libc::c_void,
//     ) -> Result<(), Error> {
//         unsafe { API::ffi_rnn_backward_bias(handle, alpha, src_desc, src_data, beta, dest_desc, dest_data) }
//     }

//  /// Computes a rnn backward function w.r.t rnn coefficient.
//     pub fn rnn_backward_rnn(
//         handle: cudnnHandle_t,
//         algo: cudnnRNNBwdRNNAlgo_t,
//         conv_desc: cudnnRNNDescriptor_t,
//         work_space: *mut ::libc::c_void,
//         work_size_in_bytes: ::libc::size_t,
//         alpha: *const ::libc::c_void,
//         src_desc: cudnnTensorDescriptor_t,
//         src_data: *const ::libc::c_void,
//         diff_desc: cudnnTensorDescriptor_t,
//         diff_data: *const ::libc::c_void,
//         beta: *const ::libc::c_void,
//         grad_desc: cudnnRNNDescriptor_t,
//         grad_data: *mut ::libc::c_void,
//     ) -> Result<(), Error> {
//         unsafe { API::ffi_rnn_backward_rnn(handle, alpha, src_desc, src_data, diff_desc, diff_data, conv_desc, algo, work_space, work_size_in_bytes, beta, grad_desc, grad_data) }
//     }

//  /// Computes a rnn backward function w.r.t the output tensor.
//     pub fn rnn_backward_data(
//         handle: cudnnHandle_t,
//         algo: cudnnRNNBwdDataAlgo_t,
//         conv_desc: cudnnRNNDescriptor_t,
//         work_space: *mut ::libc::c_void,
//         work_size_in_bytes: ::libc::size_t,
//         alpha: *const ::libc::c_void,
//         rnn_desc: cudnnRNNDescriptor_t,
//         rnn_data: *const ::libc::c_void,
//         diff_desc: cudnnTensorDescriptor_t,
//         diff_data: *const ::libc::c_void,
//         beta: *const ::libc::c_void,
//         grad_desc: cudnnTensorDescriptor_t,
//         grad_data: *mut ::libc::c_void,
//     ) -> Result<(), Error> {
//         unsafe { API::ffi_rnn_backward_data(handle, alpha, rnn_desc, rnn_data, diff_desc, diff_data, conv_desc, algo, work_space, work_size_in_bytes, beta, grad_desc, grad_data) }
//     }

//     unsafe fn ffi_create_rnn_descriptor() -> Result<cudnnRNNDescriptor_t, Error> {
//         let mut desc: cudnnRNNDescriptor_t = ::std::ptr::null_mut();
//         match cudnnCreateRNNDescriptor(&mut desc) {
//             cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(desc),
//             cudnnStatus_t::CUDNN_STATUS_ALLOC_FAILED => Err(Error::AllocFailed("The resources could not be allocated.")),
//             _ => Err(Error::Unknown("Unable to create generic CUDA cuDNN RNN Descriptor.")),
//         }
//     }

//     unsafe fn ffi_destroy_rnn_descriptor(desc: cudnnRNNDescriptor_t) -> Result<(), Error> {
//         match cudnnDestroyRNNDescriptor(desc) {
//             cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
//             _ => Err(Error::Unknown("Unable to destroy CUDA cuDNN RNN Descriptor.")),
//         }
//     }

//     unsafe fn ffi_rnn_forward(
//         handle: cudnnHandle_t,
//         alpha: *const ::libc::c_void,
//         src_desc: cudnnTensorDescriptor_t,
//         src_data: *const ::libc::c_void,
//         rnn_desc: cudnnRNNDescriptor_t,
//         rnn_data: *const ::libc::c_void,
//         conv_desc: cudnnRNNDescriptor_t,
//         algo: cudnnRNNFwdAlgo_t,
//         work_space: *mut ::libc::c_void,
//         work_size_in_bytes: ::libc::size_t,
//         beta: *const ::libc::c_void,
//         dest_desc: cudnnTensorDescriptor_t,
//         dest_data: *mut ::libc::c_void,
//     ) -> Result<(), Error> {
//         let status = cudnnRNNForward(handle, alpha, src_desc, src_data, rnn_desc, rnn_data, conv_desc, algo, work_space, work_size_in_bytes, beta, dest_desc, dest_data);
//         match status {
//             cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
//             cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("At least one of the following conditions are met: At least one of the following is NULL: `handle`, `src_desc`, `rnn_desc`, `conv_desc`, `dest_desc`, `src_data`, `alpha`, `beta`. `src_desc` and `dest_desc` have a non-matching number of dimensions. `src_desc` and `rnn_desc` have a non-matching number of dimensions. `src_desc` has fewer than three number of dimensions. `src_desc`s number of dimensions is not equal to `conv_desc`s `array_length` + 2. `src_desc` and `rnn_desc` have a non-matching number of input feature maps per image. `src_desc`, `rnn_desc` and `dest_desc` have a non-matching data type. For some spatial dimension, `rnn_desc` has a spatial size that is larger than the input spatial size (including zero-padding size).")),
//             cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("At least one of the following conditions are met: `src_desc` or `dest_desc` have negative tensor striding. `src_desc`, `rnn_desc` or `dest_desc` has a number of dimensions that is not 4 or 5. The chosen algo does not support the parameters provided; see the reference for exhaustive list of parameter support for each algo")),
//             _ => Err(Error::Unknown("Unable to compute CUDA cuDNN rnnal forward.")),
//         }
//     }

//     unsafe fn ffi_rnn_backward_bias(
//         handle: cudnnHandle_t,
//         alpha: *const ::libc::c_void,
//         src_desc: cudnnTensorDescriptor_t,
//         src_data: *const ::libc::c_void,
//         beta: *const ::libc::c_void,
//         dest_desc: cudnnTensorDescriptor_t,
//         dest_data: *mut ::libc::c_void,
//     ) -> Result<(), Error> {
//         match cudnnRNNBackwardBias(handle, alpha, src_desc, src_data, beta, dest_desc, dest_data) {
//             cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
//             cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("At least one of the following conditions are met: One of the parameters  n,h,w of the output tensor is not 1. The numbers of feature maps of the input tensor and output tensor differ. The  dataType of the two tensor descriptors are different.")),
//             _ => Err(Error::Unknown("Unable to compute CUDA cuDNN rnnal backward bias.")),
//         }
//     }

//     unsafe fn ffi_rnn_backward_rnn(
//         handle: cudnnHandle_t,
//         alpha: *const ::libc::c_void,
//         src_desc: cudnnTensorDescriptor_t,
//         src_data: *const ::libc::c_void,
//         diff_desc: cudnnTensorDescriptor_t,
//         diff_data: *const ::libc::c_void,
//         conv_desc: cudnnRNNDescriptor_t,
//         algo: cudnnRNNBwdRNNAlgo_t,
//         work_space: *mut ::libc::c_void,
//         work_size_in_bytes: ::libc::size_t,
//         beta: *const ::libc::c_void,
//         grad_desc: cudnnRNNDescriptor_t,
//         grad_data: *mut ::libc::c_void,
//     ) -> Result<(), Error> {
//         match cudnnRNNBackwardRNN(handle, alpha, src_desc, src_data, diff_desc, diff_data, conv_desc, algo, work_space, work_size_in_bytes, beta, grad_desc, grad_data) {
//             cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
//             cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("At least one of the following conditions are met: At least one of the following is NULL: `handle`, `src_desc`, `diff_desc`, `conv_desc`, `grad_desc`, `src_data`, `diff_data`, `grad_data`, `alpha`, `beta`. `src_desc` and `diff_desc` have a non-matching number of dimensions. `src_desc` and `grad_desc` have a non-matching number of dimensions. `src_desc` has fewer than three number of dimensions. `src_desc`, `diff_desc` and `grad_desc` have a non-matching data type. `src_desc` and `grad_desc` have a non-matching number of input feature maps per image.")),
//             cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("At least one of the following conditions are met: `src_desc` or `diff_desc` have negative tensor striding. `src_desc`, `diff_desc` or `grad_desc` has a number of dimensions that is not 4 or 5. The chosen algo does not support the parameters provided; see the reference for exhaustive list of parameter support for each algo")),
//             cudnnStatus_t::CUDNN_STATUS_MAPPING_ERROR => Err(Error::MappingError("An error occurs during the texture binding of the rnn data.")),
//             cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed("Execution failed to launch on GPU.")),
//             _ => Err(Error::Unknown("Unable to compute CUDA cuDNN rnnal backward rnn.")),
//         }
//     }

//     unsafe fn ffi_rnn_backward_data(
//         handle: cudnnHandle_t,
//         alpha: *const ::libc::c_void,
//         rnn_desc: cudnnRNNDescriptor_t,
//         rnn_data: *const ::libc::c_void,
//         diff_desc: cudnnTensorDescriptor_t,
//         diff_data: *const ::libc::c_void,
//         conv_desc: cudnnRNNDescriptor_t,
//         algo: cudnnRNNBwdDataAlgo_t,
//         work_space: *mut ::libc::c_void,
//         work_size_in_bytes: ::libc::size_t,
//         beta: *const ::libc::c_void,
//         grad_desc: cudnnTensorDescriptor_t,
//         grad_data: *mut ::libc::c_void,
//     ) -> Result<(), Error> {
//         match cudnnRNNBackwardData(handle, alpha, rnn_desc, rnn_data, diff_desc, diff_data, conv_desc, algo, work_space, work_size_in_bytes, beta, grad_desc, grad_data) {
//             cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
//             cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("At least one of the following conditions are met: At least one of the following is NULL: `handle`, `diff_desc`, `rnn_desc`, `conv_desc`, `grad_desc`, `diff_data`, `rnn_data`, `grad_data`, `alpha`, `beta`. `rnn_desc` and `diff_desc` have a non-matching number of dimensions. `rnn_desc` and `grad_desc` have a non-matching number of dimensions. `rnn_desc has fewer than three number of dimensions. `rnn_desc`, `grad_desc` and `diff_desc` have a non-matching data type. `rnn_desc` and `grad_desc` have a non-matching number of input feature maps per image. `diff_desc`s spatial sizes do not match with the expected size as determined by `cudnnGetRNNNdForwardOutputDim()`.")),
//             cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("At least one of the following conditions are met:  `diff_desc` or `grad_desc` have negative tensor striding. `diff_desc`, `rnn_desc` or `grad_desc` has a number of dimensions that is not 4 or 5. The chosen algo does not support the parameters provided; see the reference for exhaustive list of parameter support for each algo")),
//             cudnnStatus_t::CUDNN_STATUS_MAPPING_ERROR => Err(Error::MappingError("An error occurs during the texture binding of the rnn data or the input differential tensor data.")),
//             cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed("Execution failed to launch on GPU.")),
//             _ => Err(Error::Unknown("Unable to compute CUDA cuDNN rnnal backward data.")),
//         }
//     }
