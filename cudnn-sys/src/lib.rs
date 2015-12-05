//! Defines the Foreign Function Interface for the CUDA cuDNN API.
#![allow(non_camel_case_types)]

extern crate libc;

pub enum Struct_CUstream_st { }
pub type cudaStream_t = *mut Struct_CUstream_st;

pub enum Struct_cudnnContext { }
pub type cudnnHandle_t = *mut Struct_cudnnContext;

#[repr(C)]
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum cudnnStatus_t {
    CUDNN_STATUS_SUCCESS = 0,
    CUDNN_STATUS_NOT_INITIALIZED = 1,
    CUDNN_STATUS_ALLOC_FAILED = 2,
    CUDNN_STATUS_BAD_PARAM = 3,
    CUDNN_STATUS_INTERNAL_ERROR = 4,
    CUDNN_STATUS_INVALID_VALUE = 5,
    CUDNN_STATUS_ARCH_MISMATCH = 6,
    CUDNN_STATUS_MAPPING_ERROR = 7,
    CUDNN_STATUS_EXECUTION_FAILED = 8,
    CUDNN_STATUS_NOT_SUPPORTED = 9,
    CUDNN_STATUS_LICENSE_ERROR = 10,
}

pub enum Struct_cudnnTensorStruct { }
pub type cudnnTensorDescriptor_t = *mut Struct_cudnnTensorStruct;

pub enum Struct_cudnnConvolutionStruct { }
pub type cudnnConvolutionDescriptor_t = *mut Struct_cudnnConvolutionStruct;

pub enum Struct_cudnnPoolingStruct { }
pub type cudnnPoolingDescriptor_t = *mut Struct_cudnnPoolingStruct;

pub enum Struct_cudnnFilterStruct { }
pub type cudnnFilterDescriptor_t = *mut Struct_cudnnFilterStruct;

pub enum Struct_cudnnLRNStruct { }
pub type cudnnLRNDescriptor_t = *mut Struct_cudnnLRNStruct;

#[repr(C)]
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum cudnnDataType_t {
    CUDNN_DATA_FLOAT = 0,
    CUDNN_DATA_DOUBLE = 1,
    CUDNN_DATA_HALF = 2,
}

#[repr(C)]
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum cudnnTensorFormat_t {
    CUDNN_TENSOR_NCHW = 0,
    CUDNN_TENSOR_NHWC = 1,
}

#[repr(C)]
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum cudnnAddMode_t {
    CUDNN_ADD_IMAGE = 0,
    CUDNN_ADD_FEATURE_MAP = 1,
    CUDNN_ADD_SAME_C = 2,
    CUDNN_ADD_FULL_TENSOR = 3,
}

#[repr(C)]
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum cudnnConvolutionMode_t {
    CUDNN_CONVOLUTION = 0,
    CUDNN_CROSS_CORRELATION = 1,
}

#[repr(C)]
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum cudnnConvolutionFwdPreference_t {
    CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = 0,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = 1,
    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2,
}

#[repr(C)]
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum cudnnConvolutionFwdAlgo_t {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4,
}

#[repr(C)]
#[derive(Copy)]
pub struct Struct_Unnamed14 {
    pub algo: cudnnConvolutionFwdAlgo_t,
    pub status: cudnnStatus_t,
    pub time: ::libc::c_float,
    pub memory: ::libc::size_t,
}
impl ::std::clone::Clone for Struct_Unnamed14 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_Unnamed14 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type cudnnConvolutionFwdAlgoPerf_t = Struct_Unnamed14;

#[repr(C)]
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum cudnnConvolutionBwdFilterPreference_t {
    CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = 0,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2,
}

#[repr(C)]
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum cudnnConvolutionBwdFilterAlgo_t {
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = 3,
}

#[repr(C)]
#[derive(Copy)]
pub struct Struct_Unnamed17 {
    pub algo: cudnnConvolutionBwdFilterAlgo_t,
    pub status: cudnnStatus_t,
    pub time: ::libc::c_float,
    pub memory: ::libc::size_t,
}

impl ::std::clone::Clone for Struct_Unnamed17 {
    fn clone(&self) -> Self { *self }
}

impl ::std::default::Default for Struct_Unnamed17 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}

pub type cudnnConvolutionBwdFilterAlgoPerf_t = Struct_Unnamed17;

#[repr(C)]
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum cudnnConvolutionBwdDataPreference_t {
    CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = 0,
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = 1,
    CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = 2,
}

#[repr(C)]
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum cudnnConvolutionBwdDataAlgo_t {
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2,
}

#[repr(C)]
#[derive(Copy)]
pub struct Struct_Unnamed20 {
    pub algo: cudnnConvolutionBwdDataAlgo_t,
    pub status: cudnnStatus_t,
    pub time: ::libc::c_float,
    pub memory: ::libc::size_t,
}

impl ::std::clone::Clone for Struct_Unnamed20 {
    fn clone(&self) -> Self { *self }
}

impl ::std::default::Default for Struct_Unnamed20 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}

pub type cudnnConvolutionBwdDataAlgoPerf_t = Struct_Unnamed20;

#[repr(C)]
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum cudnnSoftmaxAlgorithm_t {
    CUDNN_SOFTMAX_FAST = 0,
    CUDNN_SOFTMAX_ACCURATE = 1,
    CUDNN_SOFTMAX_LOG = 2,
}

#[repr(C)]
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum cudnnSoftmaxMode_t {
    CUDNN_SOFTMAX_MODE_INSTANCE = 0,
    CUDNN_SOFTMAX_MODE_CHANNEL = 1,
}

#[repr(C)]
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum cudnnPoolingMode_t {
    CUDNN_POOLING_MAX = 0,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1,
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2,
}

#[repr(C)]
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum cudnnActivationMode_t {
    CUDNN_ACTIVATION_SIGMOID = 0,
    CUDNN_ACTIVATION_RELU = 1,
    CUDNN_ACTIVATION_TANH = 2,
}

pub type Enum_Unnamed25 = ::libc::c_uint;
pub const CUDNN_LRN_CROSS_CHANNEL_DIM1: ::libc::c_uint = 0;
pub type cudnnLRNMode_t = Enum_Unnamed25;

pub type Enum_Unnamed26 = ::libc::c_uint;
pub const CUDNN_DIVNORM_PRECOMPUTED_MEANS: ::libc::c_uint = 0;
pub type cudnnDivNormMode_t = Enum_Unnamed26;

extern "C" {
    pub fn cudnnGetVersion() -> ::libc::size_t;

    pub fn cudnnGetErrorString(status: cudnnStatus_t) -> *const ::libc::c_char;

    pub fn cudnnCreate(handle: *mut cudnnHandle_t) -> cudnnStatus_t;

    pub fn cudnnDestroy(handle: cudnnHandle_t) -> cudnnStatus_t;

    pub fn cudnnSetStream(handle: cudnnHandle_t, streamId: cudaStream_t) -> cudnnStatus_t;

    pub fn cudnnGetStream(handle: cudnnHandle_t, streamId: *mut cudaStream_t) -> cudnnStatus_t;

    pub fn cudnnCreateTensorDescriptor(tensorDesc: *mut cudnnTensorDescriptor_t) -> cudnnStatus_t;

    pub fn cudnnSetTensor4dDescriptor(
        tensorDesc: cudnnTensorDescriptor_t,
        format: cudnnTensorFormat_t,
        dataType: cudnnDataType_t,
        n: ::libc::c_int,
        c: ::libc::c_int,
        h: ::libc::c_int,
        w: ::libc::c_int
    ) -> cudnnStatus_t;

    pub fn cudnnSetTensor4dDescriptorEx(
        tensorDesc: cudnnTensorDescriptor_t,
        dataType: cudnnDataType_t,
        n: ::libc::c_int,
        c: ::libc::c_int,
        h: ::libc::c_int,
        w: ::libc::c_int,
        nStride: ::libc::c_int,
        cStride: ::libc::c_int,
        hStride: ::libc::c_int,
        wStride: ::libc::c_int
    ) -> cudnnStatus_t;

    pub fn cudnnGetTensor4dDescriptor(
        tensorDesc: cudnnTensorDescriptor_t,
        dataType: *mut cudnnDataType_t,
        n: *mut ::libc::c_int,
        c: *mut ::libc::c_int,
        h: *mut ::libc::c_int,
        w: *mut ::libc::c_int,
        nStride: *mut ::libc::c_int,
        cStride: *mut ::libc::c_int,
        hStride: *mut ::libc::c_int,
        wStride: *mut ::libc::c_int
    ) -> cudnnStatus_t;

    pub fn cudnnSetTensorNdDescriptor(
        tensorDesc: cudnnTensorDescriptor_t,
        dataType: cudnnDataType_t,
        nbDims: ::libc::c_int,
        dimA: *const ::libc::c_int,
        strideA: *const ::libc::c_int
    ) -> cudnnStatus_t;

    pub fn cudnnGetTensorNdDescriptor(
        tensorDesc: cudnnTensorDescriptor_t,
        nbDimsRequested: ::libc::c_int,
        dataType: *mut cudnnDataType_t,
        nbDims: *mut ::libc::c_int,
        dimA: *mut ::libc::c_int,
        strideA: *mut ::libc::c_int
    ) -> cudnnStatus_t;

    pub fn cudnnDestroyTensorDescriptor(tensorDesc: cudnnTensorDescriptor_t) -> cudnnStatus_t;

    pub fn cudnnTransformTensor(
        handle: cudnnHandle_t,
        alpha: *const ::libc::c_void,
        srcDesc: cudnnTensorDescriptor_t,
        srcData: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        destDesc: cudnnTensorDescriptor_t,
        destData: *mut ::libc::c_void
    ) -> cudnnStatus_t;

    pub fn cudnnAddTensor(
        handle: cudnnHandle_t, mode: cudnnAddMode_t,
        alpha: *const ::libc::c_void,
        biasDesc: cudnnTensorDescriptor_t,
        biasData: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        srcDestDesc: cudnnTensorDescriptor_t,
        srcDestData: *mut ::libc::c_void
    ) -> cudnnStatus_t;

    pub fn cudnnAddTensor_v3(
        handle: cudnnHandle_t,
        alpha: *const ::libc::c_void,
        biasDesc: cudnnTensorDescriptor_t,
        biasData: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        srcDestDesc: cudnnTensorDescriptor_t,
        srcDestData: *mut ::libc::c_void
    ) -> cudnnStatus_t;

    pub fn cudnnSetTensor(
        handle: cudnnHandle_t,
        srcDestDesc: cudnnTensorDescriptor_t,
        srcDestData: *mut ::libc::c_void,
        value: *const ::libc::c_void
    ) -> cudnnStatus_t;

    pub fn cudnnScaleTensor(
        handle: cudnnHandle_t,
        srcDestDesc: cudnnTensorDescriptor_t,
        srcDestData: *mut ::libc::c_void,
        alpha: *const ::libc::c_void
    ) -> cudnnStatus_t;

    pub fn cudnnCreateFilterDescriptor(filterDesc: *mut cudnnFilterDescriptor_t) -> cudnnStatus_t;

    pub fn cudnnSetFilter4dDescriptor(
        filterDesc: cudnnFilterDescriptor_t,
        dataType: cudnnDataType_t,
        k: ::libc::c_int, c: ::libc::c_int,
        h: ::libc::c_int, w: ::libc::c_int
    ) -> cudnnStatus_t;

    pub fn cudnnGetFilter4dDescriptor(
        filterDesc: cudnnFilterDescriptor_t,
        dataType: *mut cudnnDataType_t,
        k: *mut ::libc::c_int,
        c: *mut ::libc::c_int,
        h: *mut ::libc::c_int,
        w: *mut ::libc::c_int
    ) -> cudnnStatus_t;

    pub fn cudnnSetFilterNdDescriptor(
        filterDesc: cudnnFilterDescriptor_t,
        dataType: cudnnDataType_t,
        nbDims: ::libc::c_int,
        filterDimA: *const ::libc::c_int
    ) -> cudnnStatus_t;

    pub fn cudnnGetFilterNdDescriptor(
        filterDesc: cudnnFilterDescriptor_t,
        nbDimsRequested: ::libc::c_int,
        dataType: *mut cudnnDataType_t,
        nbDims: *mut ::libc::c_int,
        filterDimA: *mut ::libc::c_int
    ) -> cudnnStatus_t;

    pub fn cudnnDestroyFilterDescriptor(filterDesc: cudnnFilterDescriptor_t) -> cudnnStatus_t;

    pub fn cudnnCreateConvolutionDescriptor(convDesc: *mut cudnnConvolutionDescriptor_t) -> cudnnStatus_t;

    pub fn cudnnSetConvolution2dDescriptor(
        convDesc: cudnnConvolutionDescriptor_t,
        pad_h: ::libc::c_int,
        pad_w: ::libc::c_int,
        u: ::libc::c_int, v: ::libc::c_int,
        upscalex: ::libc::c_int,
        upscaley: ::libc::c_int,
        mode: cudnnConvolutionMode_t
    ) -> cudnnStatus_t;

    pub fn cudnnGetConvolution2dDescriptor(
        convDesc: cudnnConvolutionDescriptor_t,
        pad_h: *mut ::libc::c_int,
        pad_w: *mut ::libc::c_int,
        u: *mut ::libc::c_int,
        v: *mut ::libc::c_int,
        upscalex: *mut ::libc::c_int,
        upscaley: *mut ::libc::c_int,
        mode: *mut cudnnConvolutionMode_t
    ) -> cudnnStatus_t;

    pub fn cudnnGetConvolution2dForwardOutputDim(
        convDesc: cudnnConvolutionDescriptor_t,
        inputTensorDesc: cudnnTensorDescriptor_t,
        filterDesc: cudnnFilterDescriptor_t,
        n: *mut ::libc::c_int,
        c: *mut ::libc::c_int,
        h: *mut ::libc::c_int,
        w: *mut ::libc::c_int
    ) -> cudnnStatus_t;

    pub fn cudnnSetConvolutionNdDescriptor(
        convDesc: cudnnConvolutionDescriptor_t,
        arrayLength: ::libc::c_int,
        padA: *const ::libc::c_int,
        filterStrideA: *const ::libc::c_int,
        upscaleA: *const ::libc::c_int,
        mode: cudnnConvolutionMode_t
    ) -> cudnnStatus_t;

    pub fn cudnnGetConvolutionNdDescriptor(
        convDesc: cudnnConvolutionDescriptor_t,
        arrayLengthRequested: ::libc::c_int,
        arrayLength: *mut ::libc::c_int,
        padA: *mut ::libc::c_int,
        strideA: *mut ::libc::c_int,
        upscaleA: *mut ::libc::c_int,
        mode: *mut cudnnConvolutionMode_t
    ) -> cudnnStatus_t;

    pub fn cudnnSetConvolutionNdDescriptor_v3(
        convDesc: cudnnConvolutionDescriptor_t,
        arrayLength: ::libc::c_int,
        padA: *const ::libc::c_int,
        filterStrideA: *const ::libc::c_int,
        upscaleA: *const ::libc::c_int,
        mode: cudnnConvolutionMode_t,
        dataType: cudnnDataType_t
    ) -> cudnnStatus_t;

    pub fn cudnnGetConvolutionNdDescriptor_v3(
        convDesc: cudnnConvolutionDescriptor_t,
        arrayLengthRequested: ::libc::c_int,
        arrayLength: *mut ::libc::c_int,
        padA: *mut ::libc::c_int,
        strideA: *mut ::libc::c_int,
        upscaleA: *mut ::libc::c_int,
        mode: *mut cudnnConvolutionMode_t,
        dataType: *mut cudnnDataType_t
    ) -> cudnnStatus_t;

    pub fn cudnnGetConvolutionNdForwardOutputDim(
        convDesc: cudnnConvolutionDescriptor_t,
        inputTensorDesc: cudnnTensorDescriptor_t,
        filterDesc: cudnnFilterDescriptor_t,
        nbDims: ::libc::c_int,
        tensorOuputDimA: *mut ::libc::c_int
    ) -> cudnnStatus_t;

    pub fn cudnnDestroyConvolutionDescriptor(convDesc: cudnnConvolutionDescriptor_t) -> cudnnStatus_t;

    pub fn cudnnFindConvolutionForwardAlgorithm(
        handle: cudnnHandle_t,
        srcDesc: cudnnTensorDescriptor_t,
        filterDesc: cudnnFilterDescriptor_t,
        convDesc: cudnnConvolutionDescriptor_t,
        destDesc: cudnnTensorDescriptor_t,
        requestedAlgoCount: ::libc::c_int,
        returnedAlgoCount: *mut ::libc::c_int,
        perfResults: *mut cudnnConvolutionFwdAlgoPerf_t
    ) -> cudnnStatus_t;

    pub fn cudnnGetConvolutionForwardAlgorithm(
        handle: cudnnHandle_t,
        srcDesc: cudnnTensorDescriptor_t,
        filterDesc: cudnnFilterDescriptor_t,
        convDesc: cudnnConvolutionDescriptor_t,
        destDesc: cudnnTensorDescriptor_t,
        preference: cudnnConvolutionFwdPreference_t,
        memoryLimitInbytes: ::libc::size_t,
        algo: *mut cudnnConvolutionFwdAlgo_t
    ) -> cudnnStatus_t;

    pub fn cudnnGetConvolutionForwardWorkspaceSize(
        handle: cudnnHandle_t,
        srcDesc: cudnnTensorDescriptor_t,
        filterDesc: cudnnFilterDescriptor_t,
        convDesc: cudnnConvolutionDescriptor_t,
        destDesc: cudnnTensorDescriptor_t,
        algo: cudnnConvolutionFwdAlgo_t,
        sizeInBytes: *mut ::libc::size_t,
    ) -> cudnnStatus_t;

    pub fn cudnnConvolutionForward(
        handle: cudnnHandle_t,
        alpha: *const ::libc::c_void,
        srcDesc: cudnnTensorDescriptor_t,
        srcData: *const ::libc::c_void,
        filterDesc: cudnnFilterDescriptor_t,
        filterData: *const ::libc::c_void,
        convDesc: cudnnConvolutionDescriptor_t,
        algo: cudnnConvolutionFwdAlgo_t,
        workSpace: *mut ::libc::c_void,
        workSpaceSizeInBytes: ::libc::size_t,
        beta: *const ::libc::c_void,
        destDesc: cudnnTensorDescriptor_t,
        destData: *mut ::libc::c_void
    ) -> cudnnStatus_t;

    pub fn cudnnConvolutionBackwardBias(
        handle: cudnnHandle_t,
        alpha: *const ::libc::c_void,
        srcDesc: cudnnTensorDescriptor_t,
        srcData: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        destDesc: cudnnTensorDescriptor_t,
        destData: *mut ::libc::c_void
    ) -> cudnnStatus_t;

    pub fn cudnnFindConvolutionBackwardFilterAlgorithm(
        handle: cudnnHandle_t,
        srcDesc: cudnnTensorDescriptor_t,
        diffDesc: cudnnTensorDescriptor_t,
        convDesc: cudnnConvolutionDescriptor_t,
        gradDesc: cudnnFilterDescriptor_t,
        requestedAlgoCount: ::libc::c_int,
        returnedAlgoCount: *mut ::libc::c_int,
        perfResults: *mut cudnnConvolutionBwdFilterAlgoPerf_t
    ) -> cudnnStatus_t;

    pub fn cudnnGetConvolutionBackwardFilterAlgorithm(
        handle: cudnnHandle_t,
        srcDesc: cudnnTensorDescriptor_t,
        diffDesc: cudnnTensorDescriptor_t,
        convDesc: cudnnConvolutionDescriptor_t,
        gradDesc: cudnnFilterDescriptor_t,
        preference: cudnnConvolutionBwdFilterPreference_t,
        memoryLimitInbytes: ::libc::size_t,
        algo: *mut cudnnConvolutionBwdFilterAlgo_t
    ) -> cudnnStatus_t;

    pub fn cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle: cudnnHandle_t,
        srcDesc: cudnnTensorDescriptor_t,
        diffDesc: cudnnTensorDescriptor_t,
        convDesc: cudnnConvolutionDescriptor_t,
        gradDesc: cudnnFilterDescriptor_t,
        algo: cudnnConvolutionBwdFilterAlgo_t,
        sizeInBytes: *mut ::libc::size_t,
    ) -> cudnnStatus_t;

    pub fn cudnnConvolutionBackwardFilter_v3(
        handle: cudnnHandle_t,
        alpha: *const ::libc::c_void,
        srcDesc: cudnnTensorDescriptor_t,
        srcData: *const ::libc::c_void,
        diffDesc: cudnnTensorDescriptor_t,
        diffData: *const ::libc::c_void,
        convDesc: cudnnConvolutionDescriptor_t,
        algo: cudnnConvolutionBwdFilterAlgo_t,
        workSpace: *mut ::libc::c_void,
        workSpaceSizeInBytes: ::libc::size_t,
        beta: *const ::libc::c_void,
        gradDesc: cudnnFilterDescriptor_t,
        gradData: *mut ::libc::c_void
    ) -> cudnnStatus_t;

    pub fn cudnnConvolutionBackwardFilter(
        handle: cudnnHandle_t,
        alpha: *const ::libc::c_void,
        srcDesc: cudnnTensorDescriptor_t,
        srcData: *const ::libc::c_void,
        diffDesc: cudnnTensorDescriptor_t,
        diffData: *const ::libc::c_void,
        convDesc: cudnnConvolutionDescriptor_t,
        beta: *const ::libc::c_void,
        gradDesc: cudnnFilterDescriptor_t,
        gradData: *mut ::libc::c_void
    ) -> cudnnStatus_t;

    pub fn cudnnFindConvolutionBackwardDataAlgorithm(
        handle: cudnnHandle_t,
        filterDesc: cudnnFilterDescriptor_t,
        diffDesc: cudnnTensorDescriptor_t,
        convDesc: cudnnConvolutionDescriptor_t,
        gradDesc: cudnnTensorDescriptor_t,
        requestedAlgoCount: ::libc::c_int,
        returnedAlgoCount: *mut ::libc::c_int,
        perfResults: *mut cudnnConvolutionBwdDataAlgoPerf_t
    ) -> cudnnStatus_t;

    pub fn cudnnGetConvolutionBackwardDataAlgorithm(
        handle: cudnnHandle_t,
        filterDesc: cudnnFilterDescriptor_t,
        diffDesc: cudnnTensorDescriptor_t,
        convDesc: cudnnConvolutionDescriptor_t,
        gradDesc: cudnnTensorDescriptor_t,
        preference: cudnnConvolutionBwdDataPreference_t,
        memoryLimitInbytes: ::libc::size_t,
        algo: *mut cudnnConvolutionBwdDataAlgo_t
    ) -> cudnnStatus_t;

    pub fn cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle: cudnnHandle_t,
        filterDesc: cudnnFilterDescriptor_t,
        diffDesc: cudnnTensorDescriptor_t,
        convDesc: cudnnConvolutionDescriptor_t,
        gradDesc: cudnnTensorDescriptor_t,
        algo: cudnnConvolutionBwdDataAlgo_t,
        sizeInBytes: *mut ::libc::size_t,
    ) -> cudnnStatus_t;

    pub fn cudnnConvolutionBackwardData_v3(
        handle: cudnnHandle_t,
        alpha: *const ::libc::c_void,
        filterDesc: cudnnFilterDescriptor_t,
        filterData: *const ::libc::c_void,
        diffDesc: cudnnTensorDescriptor_t,
        diffData: *const ::libc::c_void,
        convDesc: cudnnConvolutionDescriptor_t,
        algo: cudnnConvolutionBwdDataAlgo_t,
        workSpace: *mut ::libc::c_void,
        workSpaceSizeInBytes: ::libc::size_t,
        beta: *const ::libc::c_void,
        gradDesc: cudnnTensorDescriptor_t,
        gradData: *mut ::libc::c_void
    ) -> cudnnStatus_t;

    pub fn cudnnConvolutionBackwardData(
        handle: cudnnHandle_t,
        alpha: *const ::libc::c_void,
        filterDesc: cudnnFilterDescriptor_t,
        filterData: *const ::libc::c_void,
        diffDesc: cudnnTensorDescriptor_t,
        diffData: *const ::libc::c_void,
        convDesc: cudnnConvolutionDescriptor_t,
        beta: *const ::libc::c_void,
        gradDesc: cudnnTensorDescriptor_t,
        gradData: *mut ::libc::c_void
    ) -> cudnnStatus_t;

    pub fn cudnnIm2Col(
        handle: cudnnHandle_t,
        srcDesc: cudnnTensorDescriptor_t,
        srcData: *const ::libc::c_void,
        filterDesc: cudnnFilterDescriptor_t,
        convDesc: cudnnConvolutionDescriptor_t,
        colBuffer: *mut ::libc::c_void
    ) -> cudnnStatus_t;

    pub fn cudnnSoftmaxForward(
        handle: cudnnHandle_t,
        algorithm: cudnnSoftmaxAlgorithm_t,
        mode: cudnnSoftmaxMode_t,
        alpha: *const ::libc::c_void,
        srcDesc: cudnnTensorDescriptor_t,
        srcData: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        destDesc: cudnnTensorDescriptor_t,
        destData: *mut ::libc::c_void
    ) -> cudnnStatus_t;

    pub fn cudnnSoftmaxBackward(
        handle: cudnnHandle_t,
        algorithm: cudnnSoftmaxAlgorithm_t,
        mode: cudnnSoftmaxMode_t,
        alpha: *const ::libc::c_void,
        srcDesc: cudnnTensorDescriptor_t,
        srcData: *const ::libc::c_void,
        srcDiffDesc: cudnnTensorDescriptor_t,
        srcDiffData: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        destDiffDesc: cudnnTensorDescriptor_t,
        destDiffData: *mut ::libc::c_void
    ) -> cudnnStatus_t;

    pub fn cudnnCreatePoolingDescriptor(poolingDesc: *mut cudnnPoolingDescriptor_t) -> cudnnStatus_t;

    pub fn cudnnSetPooling2dDescriptor(
        poolingDesc: cudnnPoolingDescriptor_t,
        mode: cudnnPoolingMode_t,
        windowHeight: ::libc::c_int,
        windowWidth: ::libc::c_int,
        verticalPadding: ::libc::c_int,
        horizontalPadding: ::libc::c_int,
        verticalStride: ::libc::c_int,
        horizontalStride: ::libc::c_int
    ) -> cudnnStatus_t;

    pub fn cudnnGetPooling2dDescriptor(
        poolingDesc: cudnnPoolingDescriptor_t,
        mode: *mut cudnnPoolingMode_t,
        windowHeight: *mut ::libc::c_int,
        windowWidth: *mut ::libc::c_int,
        verticalPadding: *mut ::libc::c_int,
        horizontalPadding: *mut ::libc::c_int,
        verticalStride: *mut ::libc::c_int,
        horizontalStride: *mut ::libc::c_int
    ) -> cudnnStatus_t;

    pub fn cudnnSetPoolingNdDescriptor(
        poolingDesc: cudnnPoolingDescriptor_t,
        mode: cudnnPoolingMode_t,
        nbDims: ::libc::c_int,
        windowDimA: *const ::libc::c_int,
        paddingA: *const ::libc::c_int,
        strideA: *const ::libc::c_int
    ) -> cudnnStatus_t;

    pub fn cudnnGetPoolingNdDescriptor(
        poolingDesc: cudnnPoolingDescriptor_t,
        nbDimsRequested: ::libc::c_int,
        mode: *mut cudnnPoolingMode_t,
        nbDims: *mut ::libc::c_int,
        windowDimA: *mut ::libc::c_int,
        paddingA: *mut ::libc::c_int,
        strideA: *mut ::libc::c_int
    ) -> cudnnStatus_t;

    pub fn cudnnGetPoolingNdForwardOutputDim(
        poolingDesc: cudnnPoolingDescriptor_t,
        inputTensorDesc: cudnnTensorDescriptor_t,
        nbDims: ::libc::c_int,
        outputTensorDimA: *mut ::libc::c_int
    ) -> cudnnStatus_t;

    pub fn cudnnGetPooling2dForwardOutputDim(
        poolingDesc: cudnnPoolingDescriptor_t,
        inputTensorDesc: cudnnTensorDescriptor_t,
        outN: *mut ::libc::c_int,
        outC: *mut ::libc::c_int,
        outH: *mut ::libc::c_int,
        outW: *mut ::libc::c_int
    ) -> cudnnStatus_t;

    pub fn cudnnDestroyPoolingDescriptor(poolingDesc: cudnnPoolingDescriptor_t) -> cudnnStatus_t;

    pub fn cudnnPoolingForward(
        handle: cudnnHandle_t,
        poolingDesc: cudnnPoolingDescriptor_t,
        alpha: *const ::libc::c_void,
        srcDesc: cudnnTensorDescriptor_t,
        srcData: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        destDesc: cudnnTensorDescriptor_t,
        destData: *mut ::libc::c_void
    ) -> cudnnStatus_t;

    pub fn cudnnPoolingBackward(
        handle: cudnnHandle_t,
        poolingDesc: cudnnPoolingDescriptor_t,
        alpha: *const ::libc::c_void,
        srcDesc: cudnnTensorDescriptor_t,
        srcData: *const ::libc::c_void,
        srcDiffDesc: cudnnTensorDescriptor_t,
        srcDiffData: *const ::libc::c_void,
        destDesc: cudnnTensorDescriptor_t,
        destData: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        destDiffDesc: cudnnTensorDescriptor_t,
        destDiffData: *mut ::libc::c_void
    ) -> cudnnStatus_t;

    pub fn cudnnActivationForward(
        handle: cudnnHandle_t,
        mode: cudnnActivationMode_t,
        alpha: *const ::libc::c_void,
        srcDesc: cudnnTensorDescriptor_t,
        srcData: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        destDesc: cudnnTensorDescriptor_t,
        destData: *mut ::libc::c_void
    ) -> cudnnStatus_t;

    pub fn cudnnActivationBackward(
        handle: cudnnHandle_t,
        mode: cudnnActivationMode_t,
        alpha: *const ::libc::c_void,
        srcDesc: cudnnTensorDescriptor_t,
        srcData: *const ::libc::c_void,
        srcDiffDesc: cudnnTensorDescriptor_t,
        srcDiffData: *const ::libc::c_void,
        destDesc: cudnnTensorDescriptor_t,
        destData: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        destDiffDesc: cudnnTensorDescriptor_t,
        destDiffData: *mut ::libc::c_void
    ) -> cudnnStatus_t;

    pub fn cudnnCreateLRNDescriptor(normDesc: *mut cudnnLRNDescriptor_t) -> cudnnStatus_t;

    pub fn cudnnSetLRNDescriptor(
        normDesc: cudnnLRNDescriptor_t,
        lrnN: ::libc::c_uint,
        lrnAlpha: ::libc::c_double,
        lrnBeta: ::libc::c_double,
        lrnK: ::libc::c_double
    ) -> cudnnStatus_t;

    pub fn cudnnGetLRNDescriptor(
        normDesc: cudnnLRNDescriptor_t,
        lrnN: *mut ::libc::c_uint,
        lrnAlpha: *mut ::libc::c_double,
        lrnBeta: *mut ::libc::c_double,
        lrnK: *mut ::libc::c_double
    ) -> cudnnStatus_t;

    pub fn cudnnDestroyLRNDescriptor(lrnDesc: cudnnLRNDescriptor_t) -> cudnnStatus_t;

    pub fn cudnnLRNCrossChannelForward(
        handle: cudnnHandle_t,
        normDesc: cudnnLRNDescriptor_t,
        lrnMode: cudnnLRNMode_t,
        alpha: *const ::libc::c_void,
        srcDesc: cudnnTensorDescriptor_t,
        srcData: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        destDesc: cudnnTensorDescriptor_t,
        destData: *mut ::libc::c_void
    ) -> cudnnStatus_t;

    pub fn cudnnLRNCrossChannelBackward(
        handle: cudnnHandle_t,
        normDesc: cudnnLRNDescriptor_t,
        lrnMode: cudnnLRNMode_t,
        alpha: *const ::libc::c_void,
        srcDesc: cudnnTensorDescriptor_t,
        srcData: *const ::libc::c_void,
        srcDiffDesc: cudnnTensorDescriptor_t,
        srcDiffData: *const ::libc::c_void,
        destDesc: cudnnTensorDescriptor_t,
        destData: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        destDiffDesc: cudnnTensorDescriptor_t,
        destDiffData: *mut ::libc::c_void
    ) -> cudnnStatus_t;

    pub fn cudnnDivisiveNormalizationForward(
        handle: cudnnHandle_t,
        normDesc: cudnnLRNDescriptor_t,
        mode: cudnnDivNormMode_t,
        alpha: *const ::libc::c_void,
        srcDesc: cudnnTensorDescriptor_t,
        srcData: *const ::libc::c_void,
        srcMeansData:
        *const ::libc::c_void,
        tempData: *mut ::libc::c_void,
        tempData2: *mut ::libc::c_void,
        beta: *const ::libc::c_void,
        destDesc: cudnnTensorDescriptor_t,
        destData: *mut ::libc::c_void
    ) -> cudnnStatus_t;

    pub fn cudnnDivisiveNormalizationBackward(
        handle: cudnnHandle_t,
        normDesc: cudnnLRNDescriptor_t,
        mode: cudnnDivNormMode_t,
        alpha: *const ::libc::c_void,
        srcDesc: cudnnTensorDescriptor_t,
        srcData: *const ::libc::c_void,
        srcMeansData: *const ::libc::c_void,
        srcDiffData: *const ::libc::c_void,
        tempData: *mut ::libc::c_void,
        tempData2: *mut ::libc::c_void,
        betaData: *const ::libc::c_void,
        destDataDesc: cudnnTensorDescriptor_t,
        destDataDiff: *mut ::libc::c_void,
        destMeansDiff: *mut ::libc::c_void
    ) -> cudnnStatus_t;
}
