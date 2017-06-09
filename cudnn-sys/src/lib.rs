extern crate libc;

mod generated;

pub use generated::*;

impl Default for cudnnConvolutionFwdAlgoPerf_t {
    fn default() -> Self {
        cudnnConvolutionFwdAlgoPerf_t {
            algo : cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
            status : cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED,
            time : 0.0 as f32,
            memory : 0,
            determinism : cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC,
            reserved : [0; 4usize],
        }
    }
}


impl Default for cudnnConvolutionBwdFilterAlgoPerf_t {
    fn default() -> Self {
        cudnnConvolutionBwdFilterAlgoPerf_t {
            algo : cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
            status : cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED,
            time : 0.0 as f32,
            memory : 0,
            determinism : cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC,
            reserved: [0; 4usize],
        }
    }
}


impl Default for cudnnConvolutionBwdDataAlgoPerf_t {
   fn default() -> Self {
        cudnnConvolutionBwdDataAlgoPerf_t {
            algo : cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
            status : cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED,
            time : 0.0 as f32,
            memory : 0,
            determinism : cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC,
            reserved : [0; 4usize],
        }
    }
}


