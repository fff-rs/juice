//! Provides the RNN functionality from the CUDA cuDNN API.
//!
//! Includes the RNN functionality.

use crate::ffi::*;
use crate::utils::DataType;
use crate::{Error, API};

// Workspace
impl API {
    /// This function computes the work and reserve space buffer sizes based on the RNN network geometry stored in rnnDesc, designated usage (inference or training) defined by the fMode argument, and the current RNN data dimensions (maxSeqLength, batchSize) retrieved from xDesc. When RNN data dimensions change, the cudnnGetRNNTempSpaceSizes() must be called again because RNN temporary buffer sizes are not monotonic.
    #[allow(clippy::too_many_arguments)]
    pub fn get_rnn_temp_space_size(
        handle: cudnnHandle_t,
        rnn_desc: cudnnRNNDescriptor_t,
        mode: cudnnForwardMode_t,
        x_desc: cudnnRNNDataDescriptor_t,
    ) -> Result<(usize, usize), Error> {
        let mut work_space_size: ::libc::size_t = 0;
        let mut reserved_space_size: ::libc::size_t = 0;
        unsafe {
            API::ffi_get_rnn_temp_space_size(
                handle,
                rnn_desc,
                mode,
                x_desc,
                &mut work_space_size,
                &mut reserved_space_size,
            )
        }?;
        Ok((work_space_size, reserved_space_size))
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn ffi_get_rnn_temp_space_size(
        handle: cudnnHandle_t,
        rnn_desc: cudnnRNNDescriptor_t,
        mode: cudnnForwardMode_t,
        x_desc: cudnnRNNDataDescriptor_t,
        work_space_size: *mut ::libc::size_t,
        reserved_space_size: *mut ::libc::size_t,
    ) -> Result<(), Error> {
        let status = cudnnGetRNNTempSpaceSizes(
            handle,
            rnn_desc,
            mode,
            x_desc,
            work_space_size,
            reserved_space_size,
        );
        match status {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => {
                Err(Error::BadParam("An invalid input argument was detected."))
            }
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported(
                "An incompatible or unsupported combination of input arguments was detected.",
            )),
            status => Err(Error::Unknown(
                "Unable to obtain space sized for cuDNN rnn forward.",
                status as u64,
            )),
        }
    }

    /// Returns the workspace size in byte, which are needed for the given rnnal algorithm.
    ///
    /// # Arguments
    /// * `rnn_desc` Previously initialised RNN Descriptor
    /// * `unroll_sequence_length` Length of iterations
    /// * `x_desc` An array of tensor descriptors describing the input to each recurrent iteration
    /// (one descriptor per iteration). The first dimension (batch size) of the tensors may decrease
    /// from element n to element n+1 but may not increase. For example, if you have multiple
    /// time series in a batch, they can be different lengths.
    /// This dimension is the batch size for the particular iteration of the sequence,
    /// and so it should decrease when a sequence in the batch has been terminated.
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
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("At least one of the following conditions are met: One of the parameters `x_desc`, `rnn_desc` is NULL. The tensors in `x_desc` are not of the same data type. The batch size of the tensors `x_desc` are not decreasing or staying constant.")),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("The data type used in `src_desc` is not supported for RNN.")),
            status => Err(Error::Unknown("Unable to get CUDA cuDNN RNN Forward Workspace size.", status as i32 as u64)),

        }
    }

    /// Size of Reserve Space for RNN Training [cudnnGetRNNTrainingReserveSize][1]
    /// # Arguments
    /// * `handle` Handle to cudNN Library Descriptor
    /// * `rnn_desc` Previously initialised RNN Descriptor
    /// * `seq_length` Number of iterations to unroll over - must not exceed workspace size seq_len
    /// * `x_desc` Array of tensor descriptors describing each recurrent iteration - one per element
    /// in the RNN sequence
    /// [1]: https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnGetRNNTrainingReserveSize
    pub fn get_rnn_training_reserve_size(
        handle: cudnnHandle_t,
        rnn_desc: cudnnRNNDescriptor_t,
        seq_length: ::libc::c_int,
        x_desc: Vec<cudnnTensorDescriptor_t>,
    ) -> Result<usize, Error> {
        unsafe {
            API::ffi_get_rnn_training_reserve_size(handle, rnn_desc, seq_length, x_desc.as_slice())
        }
    }
    unsafe fn ffi_get_rnn_training_reserve_size(
        handle: cudnnHandle_t,
        rnn_desc: cudnnRNNDescriptor_t,
        seq_length: ::libc::c_int,
        x_desc: &[cudnnTensorDescriptor_t],
    ) -> Result<::libc::size_t, Error> {
        let mut size: ::libc::size_t = 0;
        let size_ptr: *mut ::libc::size_t = &mut size;
        match cudnnGetRNNTrainingReserveSize(handle, rnn_desc,seq_length, x_desc.as_ptr(), size_ptr) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(size),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("At least one of the following conditions are met: One of the parameters `handle`, `x_desc`, `rnn_desc` is NULL. The tensors in `x_desc` are not of the same data type. The batch size of the tensors `x_desc` are not decreasing or staying constant.")),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("The data type used in `src_desc` is not supported for RNN.")),
            status => Err(Error::Unknown("Unable to get CUDA cuDNN RNN Training Reserve size.", status as i32 as u64)),

        }
    }
    /// cudnnGetRNNParamsSize[1]
    /// Query the amount of parameter space needed to execute the RNN for rnnDesc, given xDesc
    /// # Parameters
    /// * `handle` CUDNN Handle
    /// * `rnn_desc` Descriptor for the RNN
    /// * `x_desc` Input Tensor
    /// * `dataType` Data Type for the Input Tensor
    /// [1]: https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnGetRNNParamsSize
    pub fn get_rnn_params_size(
        handle: cudnnHandle_t,
        rnn_desc: cudnnRNNDescriptor_t,
        x_desc: cudnnTensorDescriptor_t,
        data_type: DataType,
    ) -> Result<usize, Error> {
        unsafe {
            API::ffi_get_rnn_params_size(handle, rnn_desc, x_desc, API::cudnn_data_type(data_type))
        }
    }
    unsafe fn ffi_get_rnn_params_size(
        handle: cudnnHandle_t,
        rnn_desc: cudnnRNNDescriptor_t,
        x_desc: cudnnTensorDescriptor_t,
        data_type: cudnnDataType_t,
    ) -> Result<::libc::size_t, Error> {
        let mut size: ::libc::size_t = 0;
        let size_ptr: *mut ::libc::size_t = &mut size;
        match cudnnGetRNNParamsSize(handle, rnn_desc,x_desc, size_ptr, data_type) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(size),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("One of the following; rnnDesc is invalid, x_desc is invalid, x_desc isn't fully packed, dataType & tensor Description type don't match")),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("The data type used in `rnn_desc` is not supported for RNN.")),
            status => Err(Error::Unknown("Unable to get CUDA cuDNN RNN Params Size", status as i32 as u64)),

        }
    }
}

// Descriptors
impl API {
    /// Creates a generic CUDA cuDNN RNN Descriptor.
    pub fn create_rnn_descriptor() -> Result<cudnnRNNDescriptor_t, Error> {
        unsafe { API::ffi_create_rnn_descriptor() }
    }
    unsafe fn ffi_create_rnn_descriptor() -> Result<cudnnRNNDescriptor_t, Error> {
        let mut rnn_desc: cudnnRNNDescriptor_t = ::std::ptr::null_mut();
        match cudnnCreateRNNDescriptor(&mut rnn_desc) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(rnn_desc),
            cudnnStatus_t::CUDNN_STATUS_ALLOC_FAILED => {
                Err(Error::AllocFailed("The resources could not be allocated"))
            }
            status => Err(Error::Unknown(
                "Unable create generic CUDA cuDNN RNN Descriptor",
                status as i32 as u64,
            )),
        }
    }

    /// cudnnCreateRNNDataDescriptor()
    /// https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnCreateRNNDataDescriptor
    pub fn create_rnn_data_descriptor() -> Result<cudnnRNNDataDescriptor_t, Error> {
        unsafe { API::ffi_create_rnn_data_descriptor() }
    }
    unsafe fn ffi_create_rnn_data_descriptor() -> Result<cudnnRNNDataDescriptor_t, Error> {
        let mut rnn_data_descriptor: cudnnRNNDataDescriptor_t = ::std::ptr::null_mut();
        match cudnnCreateRNNDataDescriptor(&mut rnn_data_descriptor) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(rnn_data_descriptor),
            status => Err(Error::Unknown(
                "Unable to create RNN Data Descriptor",
                status as i32 as u64,
            )),
        }
    }

    pub fn set_rnn_data_descriptor(
        rnn_data_descriptor: cudnnRNNDataDescriptor_t,
        data_type: cudnnDataType_t,
        layout: cudnnRNNDataLayout_t,
        max_sequence_length: i32,
        batch_size: i32,
        vector_size: i32,
        sequence_length_array: &[i32],
        _padding: *mut ::libc::c_void,
    ) -> Result<cudnnRNNDataDescriptor_t, Error> {
        unsafe {
            API::ffi_set_rnn_data_descriptor(
                rnn_data_descriptor,
                data_type,
                layout,
                max_sequence_length,
                batch_size,
                vector_size,
                sequence_length_array,
                ::std::ptr::null_mut() as *mut ::libc::c_void,
            )
        }
    }
    unsafe fn ffi_set_rnn_data_descriptor(
        rnn_data_descriptor: cudnnRNNDataDescriptor_t,
        data_type: cudnnDataType_t,
        layout: cudnnRNNDataLayout_t,
        max_sequence_length: i32,
        batch_size: i32,
        vector_size: i32,
        sequence_length_array: &[i32],
        padding: *mut ::libc::c_void,
    ) -> Result<cudnnRNNDataDescriptor_t, Error> {
        match cudnnSetRNNDataDescriptor(
            rnn_data_descriptor,
            data_type,
            layout,
            max_sequence_length,
            batch_size,
            vector_size,
            sequence_length_array.as_ptr(),
            padding,
        ) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(rnn_data_descriptor),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported(
                "dataType is not one of CUDNN_DATA_HALF, CUDNN_DATA_FLOAT or CUDNN_DATA_DOUBLE",
            )),
            cudnnStatus_t::CUDNN_STATUS_ALLOC_FAILED => Err(Error::AllocFailed(
                "The allocation of internal array storage has failed.",
            )),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam(
                r#"One of these have occurred:
 * rnn_data_desc is `null`.
 * Any one of `max_sequence_length`, `batch_size` or `sequence_length_array` is less than or equal to zero.
 * An element of `sequence_length_array` is less than zero or greater than `max_sequence_length`.
 * `layout` is not one of `CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED`, `CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED` or `CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED`.
"#,
            )),
            status => Err(Error::Unknown(
                "Unable to set RNN Data Descriptor",
                status as i32 as u64,
            )),
        }
    }

    /// Destroys a CUDA cuDNN RNN Descriptor.
    ///
    /// Should be called when freeing a CUDA::Descriptor to not trash up the CUDA device.
    pub fn destroy_rnn_descriptor(desc: cudnnRNNDescriptor_t) -> Result<(), Error> {
        unsafe { API::ffi_destroy_rnn_descriptor(desc) }
    }
    unsafe fn ffi_destroy_rnn_descriptor(rnn_desc: cudnnRNNDescriptor_t) -> Result<(), Error> {
        match cudnnDestroyRNNDescriptor(rnn_desc) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            status => Err(Error::Unknown(
                "Unable to destroy CUDA cuDNN Dropout Descriptor",
                status as i32 as u64,
            )),
        }
    }

    /// Initializes a generic CUDA cuDNN RNN Descriptor with specific properties.
    #[allow(clippy::too_many_arguments)]
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
        data_type: DataType,
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
            API::cudnn_data_type(data_type),
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
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
        match cudnnSetRNNDescriptor_v6(
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
            status => Err(Error::Unknown(
                "Unable to set CUDA cuDNN RNN Descriptor.",
                status as i32 as u64,
            )),
        }
    }

    /// Set RNN Matrix Math Type [cudnnSetRNNMatrixMathType][1]
    /// Required for RNN Operations[2]
    ///
    /// [1]: https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnSetRNNMatrixMathType
    /// [2]: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#tensor-ops-rnn-functions-pre-req
    pub fn set_rnn_matrix_math_type(
        rnn_desc: cudnnRNNDescriptor_t,
        math_type: cudnnMathType_t,
    ) -> Result<(), Error> {
        unsafe { API::ffi_set_rnn_matrix_math_type(rnn_desc, math_type) }
    }
    unsafe fn ffi_set_rnn_matrix_math_type(
        rnn_desc: cudnnRNNDescriptor_t,
        math_type: cudnnMathType_t,
    ) -> Result<(), Error> {
        match cudnnSetRNNMatrixMathType(rnn_desc, math_type) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("FIXME RNN")),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("FIXME RNN")),
            status => Err(Error::Unknown(
                "Unable to set CUDA cuDNN RNN Matrix Math Type.",
                status as i32 as u64,
            )),
        }
    }

    /// Set RNN Padding Model [cudnnSetRNNPaddingMode][1]
    /// This function enables or disables the padded RNN input/output for a previously created
    /// and initialized RNN descriptor. This information is required before calling
    /// the cudnnGetRNNWorkspaceSize() and cudnnGetRNNTrainingReserveSize() functions,
    /// to determine whether additional workspace and training reserve space is needed.
    /// By default, the padded RNN input/output is not enabled.
    ///
    /// [1]: https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnSetRNNPaddingMode
    pub fn set_rnn_padding_mode(
        rnn_desc: cudnnRNNDescriptor_t,
        padding_mode: cudnnRNNPaddingMode_t,
    ) -> Result<(), Error> {
        unsafe { API::ffi_set_rnn_padding_mode(rnn_desc, padding_mode) }
    }
    unsafe fn ffi_set_rnn_padding_mode(
        rnn_desc: cudnnRNNDescriptor_t,
        padding_mode: cudnnRNNPaddingMode_t,
    ) -> Result<(), Error> {
        match cudnnSetRNNPaddingMode(
            rnn_desc,
            padding_mode,
        ) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("cudnnSetRnnPaddingMode - Bad Param - Either RNN Desc is Null or paddingMode has an invalid enum (Unlikely due to Bindgen. Likely RNN Desc is somehow NULL")),
            status => Err(Error::Unknown("Unable to set CUDA cuDNN RNN Padding Mode.", status as i32 as u64)),

        }
    }
}

// Forward Training & Inference
impl API {
    /// Trains a RNN through the Forward Process
    ///
    /// # Arguments
    /// `handle` Handle to a previously created cudNN context [0]
    /// `rnn_desc` A previously initialised RNN descriptor [1]
    /// `seq_length` Number of iterations for the RNN to unroll over.
    /// `x_desc` Array of seqLength packed tensor descriptors [1]. Each descriptor should have
    /// 3D that describe the input data format to one recurrent iterator - one descriptor per
    /// RNN time-step. ```[Batch Size, Input Size, 1]```
    /// Input vectors should be column-major, so should be set
    /// ```strideA[0]=inputSize, strideA[1]=1, strideA[2]=1```
    /// `x` Data Pointer to GPU memory associated with the input.
    /// `hx_desc` Fully packed tensor descriptor for the initial hidden state of the RNN.
    /// `hx` Data pointer for initial hidden state - if null will initialize state to zero.
    /// `cx_desc` Tensor descriptor for the initial cell state for an LSTM network.
    /// `cx` Data pointer for initial cell state - if null will initialize state to zero.A
    /// `w_desc` Handle to descriptors for weights
    /// `w` Data Pointer to weights
    /// `y_desc` Output for each recurrent iteration. Second dimension should match size of the
    /// hidden layer. First dimension should match the first dimension of the tensor in input.
    /// `y` Output Memory
    /// `hy_desc` Final hidden state of the RNN
    /// `hy` Memory for final hidden state
    /// `cy_desc` Final cell state for the RNN
    /// `cy` Memory for the final cell state - can be NULL.
    /// `workspace` Data pointer to GPU memory to be used as a workspace for this call
    /// `workspace_in_bytes` Size in bytes of the provided workspace
    /// `reserve_space` Data pointer for GPU memory to be used as a reserve space for this call
    /// `reserve_space_in_bytes` Size in bytes for `reserve_space`
    /// [0] https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnHandle_t
    /// [1] https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnRNNDescriptor_t
    /// [2] https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnFilterDescriptor_t
    #[allow(clippy::too_many_arguments)]
    pub fn rnn_forward_training(
        handle: cudnnHandle_t,
        rnn_desc: cudnnRNNDescriptor_t,
        seq_length: ::libc::c_int,
        x_desc: Vec<cudnnTensorDescriptor_t>,
        x: *const ::libc::c_void,
        hx_desc: cudnnTensorDescriptor_t,
        hx: *const ::libc::c_void,
        cx_desc: cudnnTensorDescriptor_t,
        cx: *const ::libc::c_void,
        w_desc: cudnnFilterDescriptor_t,
        w: *const ::libc::c_void,
        y_desc: Vec<cudnnTensorDescriptor_t>,
        y: *mut ::libc::c_void,
        hy_desc: cudnnTensorDescriptor_t,
        hy: *mut ::libc::c_void,
        cy_desc: cudnnTensorDescriptor_t,
        cy: *mut ::libc::c_void,
        workspace: *mut ::libc::c_void,
        workspace_size_in_bytes: usize,
        reserve: *mut ::libc::c_void,
        reserve_size_in_bytes: usize,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_rnn_forward_training(
                handle,
                rnn_desc,
                seq_length,
                x_desc.as_slice(),
                x,
                hx_desc,
                hx,
                cx_desc,
                cx,
                w_desc,
                w,
                y_desc.as_slice(),
                y,
                hy_desc,
                hy,
                cy_desc,
                cy,
                workspace,
                workspace_size_in_bytes,
                reserve,
                reserve_size_in_bytes,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn ffi_rnn_forward_training(
        handle: cudnnHandle_t,
        rnn_desc: cudnnRNNDescriptor_t,
        seq_length: ::libc::c_int,
        x_desc: &[cudnnTensorDescriptor_t],
        x: *const ::libc::c_void,
        hx_desc: cudnnTensorDescriptor_t,
        hx: *const ::libc::c_void,
        cx_desc: cudnnTensorDescriptor_t,
        cx: *const ::libc::c_void,
        w_desc: cudnnFilterDescriptor_t,
        w: *const ::libc::c_void,
        y_desc: &[cudnnTensorDescriptor_t],
        y: *mut ::libc::c_void,
        hy_desc: cudnnTensorDescriptor_t,
        hy: *mut ::libc::c_void,
        cy_desc: cudnnTensorDescriptor_t,
        cy: *mut ::libc::c_void,
        workspace: *mut ::libc::c_void,
        work_space_size_in_bytes: usize,
        reserve_space: *mut ::libc::c_void,
        reserve_space_size_in_bytes: usize,
    ) -> Result<(), Error> {
        let status = cudnnRNNForwardTraining(
            handle,
            rnn_desc,
            seq_length,
            x_desc.as_ptr(),
            x,
            hx_desc,
            hx,
            cx_desc,
            cx,
            w_desc,
            w,
            y_desc.as_ptr(),
            y,
            hy_desc,
            hy,
            cy_desc,
            cy,
            workspace,
            work_space_size_in_bytes,
            reserve_space,
            reserve_space_size_in_bytes,
        );
        match status {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("At least one of the following conditions was met: rnnDesc is invalid, hx_desc, w_desc, hy_desc, cy_desc, or one of the x_desc or y_desc is invalid. The descriptors for x_desc, cx_desc, _hx_desc, w_desc, y_desc, hy_desc, cy_desc have incorrect strides/diemnsions. Workspace size is too small. Reserve space size is too small.")),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("At least one of the following conditions are met: `src_desc` or `dest_desc` have negative tensor striding. `src_desc`, `rnn_desc` or `dest_desc` has a number of dimensions that is not 4 or 5. The chosen algo does not support the parameters provided; see the reference for exhaustive list of parameter support for each algo")),
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed("The function failed to launch on the GPU.")),
            cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE => Err(Error::InvalidValue("cudnnSetPersistentRNNPlan() was not called prior to the current function when CUDNN_RNN_ALGO_PERSIST_DYNAMIC was selected in the RNN descriptor.")),
            cudnnStatus_t::CUDNN_STATUS_ALLOC_FAILED => Err(Error::AllocFailed("The function was unable to allocate memory.")),
             status => Err(Error::Unknown("Unable to compute CUDA cuDNN rnn forward.", status as u64)),
        }
    }

    /// Execute a RNN without Training
    /// This routine executes the recurrent neural network described by rnnDesc with inputs x, hx,
    /// and cx, weights w and outputs y, hy, and cy. workspace is required for intermediate storage.
    /// This function does not store intermediate data required for training;
    /// cudnnRNNForwardTraining() should be used for that purpose
    ///
    /// # Arguments
    /// `handle` Handle to a previously created cudNN context [0]
    /// `rnn_desc` A previously initialised RNN descriptor [1]
    /// `seq_length` Number of iterations for the RNN to unroll over.
    /// `x_desc` Array of seqLength packed tensor descriptors [1]. Each descriptor should have
    /// 3D that describe the input data format to one recurrent iterator - one descriptor per
    /// RNN time-step. ```[Batch Size, Input Size, 1]```
    /// Input vectors should be column-major, so should be set
    /// strideA 0 = inputSize, strideA 1 = 1, strideA 2 =1
    /// `x` Data Pointer to GPU memory associated with the input.
    /// `hx_desc` Fully packed tensor descriptor for the initial hidden state of the RNN.
    /// `hx` Data pointer for initial hidden state - if null will initialize state to zero.
    /// `cx_desc` Tensor descriptor for the initial cell state for an LSTM network.
    /// `cx` Data pointer for initial cell state - if null will initialize state to zero.A
    /// `w_desc` Handle to descriptors for weights
    /// `w` Data Pointer to weights
    /// `y_desc` Output for each recurrent iteration. Second dimension should match size of the
    /// hidden layer. First dimension should match the first dimension of the tensor in input.
    /// `y` Output Memory
    /// `hy_desc` Final hidden state of the RNN
    /// `hy` Memory for final hidden state
    /// `cy_desc` Final cell state for the RNN
    /// `cy` Memory for the final cell state - can be NULL.
    /// `workspace` Data pointer to GPU memory to be used as a workspace for this call
    /// `workspace_in_bytes` Size in bytes of the provided workspace
    /// [0] https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnHandle_t
    /// [1] https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnRNNDescriptor_t
    #[allow(clippy::too_many_arguments)]
    pub fn rnn_forward_inference(
        handle: cudnnHandle_t,
        rnn_desc: cudnnRNNDescriptor_t,
        seq_length: ::libc::c_int,
        x_desc: *const cudnnTensorDescriptor_t,
        x: *mut ::libc::c_void,
        hx_desc: cudnnTensorDescriptor_t,
        hx: *mut ::libc::c_void,
        cx_desc: cudnnTensorDescriptor_t,
        cx: *mut ::libc::c_void,
        w_desc: cudnnFilterDescriptor_t,
        w: *mut ::libc::c_void,
        y_desc: *const cudnnTensorDescriptor_t,
        y: *mut ::libc::c_void,
        hy_desc: cudnnTensorDescriptor_t,
        hy: *mut ::libc::c_void,
        cy_desc: cudnnTensorDescriptor_t,
        cy: *mut ::libc::c_void,
        work_space: *mut ::libc::c_void,
        work_size_in_bytes: ::libc::size_t,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_rnn_forward_inference(
                handle,
                rnn_desc,
                seq_length,
                x_desc,
                x,
                hx_desc,
                hx,
                cx_desc,
                cx,
                w_desc,
                w,
                y_desc,
                y,
                hy_desc,
                hy,
                cy_desc,
                cy,
                work_space,
                work_size_in_bytes,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn ffi_rnn_forward_inference(
        handle: cudnnHandle_t,
        rnn_desc: cudnnRNNDescriptor_t,
        seq_length: ::libc::c_int,
        x_desc: *const cudnnTensorDescriptor_t,
        x: *const ::libc::c_void,
        hx_desc: cudnnTensorDescriptor_t,
        hx: *const ::libc::c_void,
        cx_desc: cudnnTensorDescriptor_t,
        cx: *const ::libc::c_void,
        w_desc: cudnnFilterDescriptor_t,
        w: *const ::libc::c_void,
        y_desc: *const cudnnTensorDescriptor_t,
        y: *mut ::libc::c_void,
        hy_desc: cudnnTensorDescriptor_t,
        hy: *mut ::libc::c_void,
        cy_desc: cudnnTensorDescriptor_t,
        cy: *mut ::libc::c_void,
        workspace: *mut ::libc::c_void,
        work_space_size_in_bytes: usize,
    ) -> Result<(), Error> {
        let status = cudnnRNNForwardInference(
            handle,
            rnn_desc,
            seq_length,
            x_desc,
            x,
            hx_desc,
            hx,
            cx_desc,
            cx,
            w_desc,
            w,
            y_desc,
            y,
            hy_desc,
            hy,
            cy_desc,
            cy,
            workspace,
            work_space_size_in_bytes,
        );
        match status {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("At least one of the following conditions are met: At least one of the following is NULL: `handle`, `src_desc`, `rnn_desc`, `conv_desc`, `dest_desc`, `src_data`, `alpha`, `beta`. `src_desc` and `dest_desc` have a non-matching number of dimensions. `src_desc` and `rnn_desc` have a non-matching number of dimensions. `src_desc` has fewer than three number of dimensions. `src_desc`s number of dimensions is not equal to `conv_desc`s `array_length` + 2. `src_desc` and `rnn_desc` have a non-matching number of input feature maps per image. `src_desc`, `rnn_desc` and `dest_desc` have a non-matching data type. For some spatial dimension, `rnn_desc` has a spatial size that is larger than the input spatial size (including zero-padding size).")),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("At least one of the following conditions are met: `src_desc` or `dest_desc` have negative tensor striding. `src_desc`, `rnn_desc` or `dest_desc` has a number of dimensions that is not 4 or 5. The chosen algo does not support the parameters provided; see the reference for exhaustive list of parameter support for each algo")),
            status => Err(Error::Unknown("Unable to compute CUDA cuDNN rnnal forward.", status as i32 as u64)),

        }
    }
}

// Backward Training, Bias, Weights, and IInference
impl API {
    /// CUDNN Rnn Backward Data
    /// This routine executes the recurrent neural network described by rnnDesc with output
    /// gradients dy, dhy, and dhc, weights w and input gradients dx, dhx, and dcx.
    /// Workspace is required for intermediate storage.
    /// The data in reserveSpace must have previously been generated by cudnnRNNForwardTraining().
    /// The same reserveSpace data must be used for future calls to cudnnRNNBackwardWeights()
    /// if they execute on the same input data.
    ///
    /// # Arguments
    /// `handle` Handle to a previously created [cudNN context][0]
    /// `rnn_desc` A previously initialised [RNN descriptor][1]
    /// `seq_length` Number of iterations for the RNN to unroll over.
    /// `y_desc` Array of packed [tensor descriptors][1] describing the *output* from each recurrent
    /// iteration.
    /// `y` Data pointer to GPU memory for output at each iteration
    /// `dy_desc` Array of packed [tensor descriptors][1] describing the *gradient* at the output
    /// from each recurrent iteration.
    /// `dy` Data pointer to GPU memory for gradient at output iterations
    /// `dhy_desc` Array of packed [tensor descriptors][1] describing the *gradients* at the final *hidden*
    /// state of the RNN.
    /// `dhy` Data pointer to GPU memory for gradient at the final hidden state of the network.
    /// If this is a NULL pointer, the gradients at the final hidden state of the network will be
    /// initialised to zero.
    /// `dcy_desc` Array of packed [tensor descriptors][1] describing the *gradients* at the final *cell*
    /// state of the RNN.
    /// `dcy` Data pointer to GPU memory for gradients at the final cell state of the RNN.
    /// `w_desc` Handle to a previously initialized filter descriptor for the weights in the RNN
    /// `w` Data pointer to GPU memory for the filter descriptor for the weights.
    /// `hx_desc` Fully packed tensor descriptor for the initial hidden state of the RNN.
    /// `hx` Data pointer for initial hidden state - if null will initialize state to zero.
    /// `cx_desc` Tensor descriptor for the initial cell state for an LSTM network.
    /// `cx` Data pointer for initial cell state - if null will initialize state to zero.
    /// `dx_desc` Array of fully packed tensor descriptors for the gradient at the input of each
    /// iteration.
    /// `dx` Data pointer for the gradient of the input of each recurrent iteration.
    /// `dhx_desc` Fully packed tensor for the gradient of the initial hidden state of the RNN.
    /// `dhx` Data pointer for gradient of the initial hidden state of the RNN.
    /// `workspace` Data pointer to GPU memory to be used as a workspace for this call
    /// `workspace_in_bytes` Size in bytes of the provided workspace
    /// `reserve_space` Data pointer for GPU memory to be used as a reserve space for this call
    /// `reserve_space_in_bytes` Size in bytes for `reserve_space`
    /// [0]:https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnHandle_t
    /// [1]:https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnRNNDescriptor_t
    #[allow(clippy::too_many_arguments)]
    pub fn rnn_backward_data(
        handle: cudnnHandle_t,
        rnn_desc: cudnnRNNDescriptor_t,
        seq_length: ::libc::c_int,
        y_desc: *const cudnnTensorDescriptor_t,
        y: *const ::libc::c_void,
        dy_desc: *const cudnnTensorDescriptor_t,
        dy: *const ::libc::c_void,
        dhy_desc: cudnnTensorDescriptor_t,
        dhy: *const ::libc::c_void,
        dcy_desc: cudnnTensorDescriptor_t,
        dcy: *const ::libc::c_void,
        w_desc: cudnnFilterDescriptor_t,
        w: *const ::libc::c_void,
        hx_desc: cudnnTensorDescriptor_t,
        hx: *const ::libc::c_void,
        cx_desc: cudnnTensorDescriptor_t,
        cx: *const ::libc::c_void,
        dx_desc: *const cudnnTensorDescriptor_t,
        dx: *mut ::libc::c_void,
        dhx_desc: cudnnTensorDescriptor_t,
        dhx: *mut ::libc::c_void,
        dcx_desc: cudnnTensorDescriptor_t,
        dcx: *mut ::libc::c_void,
        workspace: *mut ::libc::c_void,
        workspace_size_in_bytes: usize,
        reserve_space: *mut ::libc::c_void,
        reserve_space_size_in_bytes: usize,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_rnn_backward_data(
                handle,
                rnn_desc,
                seq_length,
                y_desc,
                y,
                dy_desc,
                dy,
                dhy_desc,
                dhy,
                dcy_desc,
                dcy,
                w_desc,
                w,
                hx_desc,
                hx,
                cx_desc,
                cx,
                dx_desc,
                dx,
                dhx_desc,
                dhx,
                dcx_desc,
                dcx,
                workspace,
                workspace_size_in_bytes,
                reserve_space,
                reserve_space_size_in_bytes,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn ffi_rnn_backward_data(
        handle: cudnnHandle_t,
        rnn_desc: cudnnRNNDescriptor_t,
        seq_length: ::libc::c_int,
        y_desc: *const cudnnTensorDescriptor_t,
        y: *const ::libc::c_void,
        dy_desc: *const cudnnTensorDescriptor_t,
        dy: *const ::libc::c_void,
        dhy_desc: cudnnTensorDescriptor_t,
        dhy: *const ::libc::c_void,
        dcy_desc: cudnnTensorDescriptor_t,
        dcy: *const ::libc::c_void,
        w_desc: cudnnFilterDescriptor_t,
        w: *const ::libc::c_void,
        hx_desc: cudnnTensorDescriptor_t,
        hx: *const ::libc::c_void,
        cx_desc: cudnnTensorDescriptor_t,
        cx: *const ::libc::c_void,
        dx_desc: *const cudnnTensorDescriptor_t,
        dx: *mut ::libc::c_void,
        dhx_desc: cudnnTensorDescriptor_t,
        dhx: *mut ::libc::c_void,
        dcx_desc: cudnnTensorDescriptor_t,
        dcx: *mut ::libc::c_void,
        workspace: *mut ::libc::c_void,
        workspace_size_in_bytes: usize,
        reserve_space: *mut ::libc::c_void,
        reserve_space_size_in_bytes: usize,
    ) -> Result<(), Error> {
        match cudnnRNNBackwardData(
            handle,
            rnn_desc,
            seq_length,
            y_desc,
            y,
            dy_desc,
            dy,
            dhy_desc,
            dhy,
            dcy_desc,
            dcy,
            w_desc,
            w,
            hx_desc,
            hx,
            cx_desc,
            cx,
            dx_desc,
            dx,
            dhx_desc,
            dhx,
            dcx_desc,
            dcx,
            workspace,
            workspace_size_in_bytes,
            reserve_space,
            reserve_space_size_in_bytes
        ) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("At least one of the following conditions are met: At least one of the following is NULL: `handle`, `diff_desc`, `rnn_desc`, `conv_desc`, `grad_desc`, `diff_data`, `rnn_data`, `grad_data`, `alpha`, `beta`. `rnn_desc` and `diff_desc` have a non-matching number of dimensions. `rnn_desc` and `grad_desc` have a non-matching number of dimensions. `rnn_desc has fewer than three number of dimensions. `rnn_desc`, `grad_desc` and `diff_desc` have a non-matching data type. `rnn_desc` and `grad_desc` have a non-matching number of input feature maps per image. `diff_desc`s spatial sizes do not match with the expected size as determined by `cudnnGetRNNNdForwardOutputDim()`.")),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("At least one of the following conditions are met:  `diff_desc` or `grad_desc` have negative tensor striding. `diff_desc`, `rnn_desc` or `grad_desc` has a number of dimensions that is not 4 or 5. The chosen algo does not support the parameters provided; see the reference for exhaustive list of parameter support for each algo")),
            cudnnStatus_t::CUDNN_STATUS_MAPPING_ERROR => Err(Error::MappingError("An error occurs during the texture binding of the rnn data or the input differential tensor data.")),
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed("Execution failed to launch on GPU.")),
            status => Err(Error::Unknown("Unable to compute CUDA cuDNN rnnal backward data.", status as i32 as u64)),

        }
    }

    /// CUDNN Rnn Backward Weights
    /// This routine accumulates weight gradients `dw` from the recurrent neural network described by
    /// rnnDesc with inputs `x`, `hx` and outputs `y`. The mode of operation in this case is additive,
    /// the weight gradients calculated will be added to those already existing in `dw`.
    /// Workspace is required for intermediate storage.
    /// The data in reserveSpace must have previously been generated by cudnnRNNBackwardData().
    ///
    /// # Arguments
    /// `handle` Handle to a previously created [cudNN context][0]
    /// `rnn_desc` A previously initialised [RNN descriptor][1]
    /// `seq_length` Number of iterations for the RNN to unroll over.
    /// `x_desc` Array of packed tensor descriptors.
    /// `x` Data pointer for Input
    /// `hx_desc` Fully packed tensor descriptor for the initial hidden state of the RNN.
    /// `hx` Data pointer for initial hidden state - if null will initialize state to zero.
    /// `y_desc` Array of packed [tensor descriptors][1] describing the *output* from each recurrent
    /// iteration.
    /// `y` Data pointer to GPU memory for output at each iteration
    /// `dw_desc` Handle to previously initialized filter descriptor for the gradient of the
    /// weights.
    /// `dw` Data pointer to GPU memory for the descriptor of the gradient of the weights.
    /// `workspace` Data pointer to GPU memory to be used as a workspace for this call
    /// `workspace_in_bytes` Size in bytes of the provided workspace
    /// `reserve_space` Data pointer for GPU memory to be used as a reserve space for this call
    /// `reserve_space_in_bytes` Size in bytes for `reserve_space`
    /// [0]:https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnHandle_t
    /// [1]:https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnRNNDescriptor_t
    #[allow(clippy::too_many_arguments)]
    pub fn rnn_backward_weights(
        handle: cudnnHandle_t,
        rnn_desc: cudnnRNNDescriptor_t,
        seq_length: ::libc::c_int,
        x_desc: *const cudnnTensorDescriptor_t,
        x: *const ::libc::c_void,
        hx_desc: cudnnTensorDescriptor_t,
        hx: *const ::libc::c_void,
        y_desc: *const cudnnTensorDescriptor_t,
        y: *const ::libc::c_void,
        workspace: *const ::libc::c_void,
        work_space_size_in_bytes: usize,
        dw_desc: cudnnFilterDescriptor_t,
        dw: *mut ::libc::c_void,
        reserve_space: *const ::libc::c_void,
        reserve_space_size_in_bytes: usize,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_rnn_backward_weights(
                handle,
                rnn_desc,
                seq_length,
                x_desc,
                x,
                hx_desc,
                hx,
                y_desc,
                y,
                workspace,
                work_space_size_in_bytes,
                dw_desc,
                dw,
                reserve_space,
                reserve_space_size_in_bytes,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn ffi_rnn_backward_weights(
        handle: cudnnHandle_t,
        rnn_desc: cudnnRNNDescriptor_t,
        seq_length: ::libc::c_int,
        x_desc: *const cudnnTensorDescriptor_t,
        x: *const ::libc::c_void,
        hx_desc: cudnnTensorDescriptor_t,
        hx: *const ::libc::c_void,
        y_desc: *const cudnnTensorDescriptor_t,
        y: *const ::libc::c_void,
        workspace: *const ::libc::c_void,
        work_space_size_in_bytes: usize,
        dw_desc: cudnnFilterDescriptor_t,
        dw: *mut ::libc::c_void,
        reserve_space: *const ::libc::c_void,
        reserve_space_size_in_bytes: usize,
    ) -> Result<(), Error> {
        match cudnnRNNBackwardWeights(
            handle,
            rnn_desc,
            seq_length,
            x_desc,
            x,
            hx_desc,
            hx,
            y_desc,
            y,
            workspace,
            work_space_size_in_bytes,
            dw_desc,
            dw,
            reserve_space,
            reserve_space_size_in_bytes,
        ) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("At least one of the following conditions are met: At least one of the following is NULL: `handle`, `src_desc`, `diff_desc`, `conv_desc`, `grad_desc`, `src_data`, `diff_data`, `grad_data`, `alpha`, `beta`. `src_desc` and `diff_desc` have a non-matching number of dimensions. `src_desc` and `grad_desc` have a non-matching number of dimensions. `src_desc` has fewer than three number of dimensions. `src_desc`, `diff_desc` and `grad_desc` have a non-matching data type. `src_desc` and `grad_desc` have a non-matching number of input feature maps per image.")),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("At least one of the following conditions are met: `src_desc` or `diff_desc` have negative tensor striding. `src_desc`, `diff_desc` or `grad_desc` has a number of dimensions that is not 4 or 5. The chosen algo does not support the parameters provided; see the reference for exhaustive list of parameter support for each algo")),
            cudnnStatus_t::CUDNN_STATUS_MAPPING_ERROR => Err(Error::MappingError("An error occurs during the texture binding of the rnn data.")),
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed("Execution failed to launch on GPU.")),
            status => Err(Error::Unknown("Unable to compute CUDA cuDNN rnnal backward rnn.", status as i32 as u64)),

        }
    }
}
