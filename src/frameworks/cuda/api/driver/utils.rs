//! Provides Cuda Driver API utility functionality.

use super::{API, Error};
use super::ffi::*;

impl API {
    /// Initialize the Cuda Driver API.
    ///
    /// must be called before any other function from the driver API.
    pub fn init() -> Result<(), Error> {
        Ok(try!( unsafe { API::ffi_init() }))
    }

    unsafe fn ffi_init() -> Result<(), Error> {
        match cuInit(0u32) {
            CUresult::CUDA_SUCCESS => Ok(()),
            CUresult::CUDA_ERROR_INVALID_VALUE => Err(Error::InvalidValue("Invalid value provided.")),
            CUresult::CUDA_ERROR_INVALID_DEVICE => Err(Error::InvalidDevice("Invalid device.")),
            CUresult::CUDA_ERROR_NO_DEVICE => Err(Error::NoDevice("Unable to find a CUDA device. Try run `nvidia-smi` on your console.")),
            _ => Err(Error::Unknown("Unable to initialze the Cuda Driver API.")),
        }
    }
}
