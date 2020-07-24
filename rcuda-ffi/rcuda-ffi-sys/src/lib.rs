use std::fmt;

#[derive(Debug, Clone)]
struct CuFfiError {
    msg: String
}

impl fmt::Display for CuFfiError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad(&self.msg)
    }
}

impl CuFfiError {
    fn new(msg: String) -> CuFfiError {
        CuFfiError { msg }
    }
}

impl std::error::Error for CuFfiError {}

extern "C" {
    pub(crate) fn cuGather(
        embedding_dimension: usize,
        phrase_length: usize,
        vocab_size: usize,
        batch_size: usize,
        src_ptr: *const libc::c_void,
        weight_ptr: *const libc::c_void,
        dest_ptr: *mut libc::c_void,
    ) -> usize;

    pub(crate) fn cuBatchStridedSum(
        input_ptr: *const libc::c_void,
        output_ptr: *mut libc::c_void,
        batch_size: usize,
        rows: usize,
        cols: usize,
    ) -> usize;
}

// Functions implemented in Rust.
pub fn ffi_gather(
    embedding_dimension: usize,
    phrase_length: usize,
    vocab_size: usize,
    batch_size: usize,
    src_ptr: *const libc::c_void,
    weight_ptr: *const libc::c_void,
    dest_ptr: *mut libc::c_void,
) -> Result<usize, Box<dyn std::error::Error>> {
    unsafe {
        let cu_status = cuGather(
            embedding_dimension,
            phrase_length,
            vocab_size,
            batch_size,
            src_ptr,
            weight_ptr,
            dest_ptr,
        );
        match cu_status {
            0 => Ok(cu_status),
            _ => Err(Box::new(CuFfiError::new("Gather failed.".to_string())))
        }
    }
}

pub fn ffi_batch_strided_sum(
    input_ptr: *const libc::c_void,
    output_ptr: *mut libc::c_void,
    batch_size: usize,
    rows: usize,
    cols: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    unsafe {
        let cu_status = cuBatchStridedSum(
            input_ptr,
            output_ptr,
            batch_size,
            rows,
            cols,
        );
        match cu_status {
            0 => Ok(cu_status),
            _ => Err(Box::new(CuFfiError::new("Batch Strided Sum failed.".to_string())))
        }
    }
}
