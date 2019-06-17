extern crate libc;

mod generated;

pub use crate::generated::*;

impl Default for cudnnConvolutionFwdAlgoPerf_t {
    fn default() -> Self {
        Self {
            algo: cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
            status: cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED,
            time: 0.0 as f32,
            memory: 0,
            determinism: cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC,
            mathType: cudnnMathType_t::CUDNN_DEFAULT_MATH,
            reserved: [0; 3usize],
        }
    }
}

impl Default for cudnnConvolutionBwdFilterAlgoPerf_t {
    fn default() -> Self {
        Self {
            algo: cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
            status: cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED,
            time: 0.0 as f32,
            memory: 0,
            determinism: cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC,
            mathType: cudnnMathType_t::CUDNN_DEFAULT_MATH,
            reserved: [0; 3usize],
        }
    }
}

impl Default for cudnnConvolutionBwdDataAlgoPerf_t {
    fn default() -> Self {
        Self {
            algo: cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
            status: cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED,
            time: 0.0 as f32,
            memory: 0,
            determinism: cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC,
            mathType: cudnnMathType_t::CUDNN_DEFAULT_MATH,
            reserved: [0; 3usize],
        }
    }
}
