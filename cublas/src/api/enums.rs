use ffi::*;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum PointerMode {
    Host,
    Device
}

impl PointerMode {
    pub fn from_c(in_mode: cublasPointerMode_t) -> PointerMode {
        match in_mode {
            cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST => PointerMode::Host,
            cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE => PointerMode::Device,
        }
    }

    pub fn as_c(&self) -> cublasPointerMode_t {
        match *self {
            PointerMode::Host => cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
            PointerMode::Device => cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE,
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Operation {
    NoTrans,
    Trans,
    ConjTrans,
}

impl Operation {
    pub fn from_c(in_mode: cublasOperation_t) -> Operation {
        match in_mode {
            cublasOperation_t::CUBLAS_OP_N => Operation::NoTrans,
            cublasOperation_t::CUBLAS_OP_T => Operation::Trans,
            cublasOperation_t::CUBLAS_OP_C => Operation::ConjTrans,
        }
    }

    pub fn as_c(&self) -> cublasOperation_t {
        match *self {
            Operation::NoTrans => cublasOperation_t::CUBLAS_OP_N,
            Operation::Trans => cublasOperation_t::CUBLAS_OP_T,
            Operation::ConjTrans => cublasOperation_t::CUBLAS_OP_C,
        }
    }
}

// TODO: cublasFillMode_t
// TODO: cublasDiagType_t
// TODO: cublasSideMode_t
// TODO: cublasAtomicsMode_t
// TODO: cublasDataType_t
