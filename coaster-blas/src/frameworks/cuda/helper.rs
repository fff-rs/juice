/// Those macros should be removed when read()/read_only()/write() are refactored
/// to return typed memory. For now they remove a lot of visual clutter and
/// lessen probability of stupid mistakes.
macro_rules! read {
    ($x:ident, $slf:ident) => {
        $x.read($slf.device())?
    };
}

/// acquire a tensor as read write
macro_rules! read_write {
    ($x:ident, $slf:ident) => {
        $x.read_write($slf.device())?
    };
}

/// acquire a tensor as write only
macro_rules! write_only {
    ($x:ident, $slf:ident) => {
        $x.write_only($slf.device())?
    };
}

/// trans! cannot be inlined into macros above, because `$mem` would become
/// intermidiate variable and `*mut $t` will outlive it.
macro_rules! trans {
    ($mem:ident, $t:ident) => {
        *$mem.id_c() as *mut f32
        //::std::mem::transmute::<u64, *mut $t>(*$mem.id_c()) }
    };
}

/// execute something and map the error as a plugin error
macro_rules! exec {
    ($name:ident, $f:expr) => ({
        let res = $f;
        res.map_err(|e| {
            log::debug!("Unable to execute operation {}: {:?}", stringify!($name), e);
            PluginError::Operation(
                stringify!(Unable to execute operation $name)
            ).into()
        })
    })
}

/// asum with cuda
#[macro_export]
macro_rules! iblas_asum_for_cuda {
    ($t:ident) => {
        fn asum(
            &self,
            x: &SharedTensor<$t>,
            result: &mut SharedTensor<$t>,
        ) -> Result<(), ::coaster::error::Error> {
            let n = x.desc().size() as i32;
            let x_mem = read!(x, self);
            let r_mem = write_only!(result, self);

            let ctx : &cublas::Context = self.framework().cublas();
            exec!(
                asum,
                (*ctx).asum(trans!(x_mem, $t), trans!(r_mem, $t), n, None)
            )
        }
    };
}

/// axpy with cuda
#[macro_export]
macro_rules! iblas_axpy_for_cuda {
    ($t:ident) => {
        fn axpy(
            &self,
            a: &SharedTensor<$t>,
            x: &SharedTensor<$t>,
            y: &mut SharedTensor<$t>,
        ) -> Result<(), ::coaster::error::Error> {
            let n = x.desc().size() as i32;
            let a_mem = read!(a, self);
            let x_mem = read!(x, self);
            let y_mem = read_write!(y, self);

            let ctx = (*self.framework()).cublas();
            exec!(
                axpy,
                ctx.axpy(
                    trans!(a_mem, $t),
                    trans!(x_mem, $t),
                    trans!(y_mem, $t),
                    n,
                    None,
                    None
                )
            )
        }
    };
}

/// copy for cuda
#[macro_export]
macro_rules! iblas_copy_for_cuda {
    ($t:ident) => {
        fn copy(
            &self,
            x: &SharedTensor<$t>,
            y: &mut SharedTensor<$t>,
        ) -> Result<(), ::coaster::error::Error> {
            let n = x.desc().size() as i32;
            let x_mem = read!(x, self);
            let y_mem = write_only!(y, self);

            let ctx = (*self.framework()).cublas();
            exec!(
                copy,
                ctx.copy(trans!(x_mem, $t), trans!(y_mem, $t), n, None, None)
            )
        }
    };
}

/// nrm2 for cuda
#[macro_export]
macro_rules! iblas_nrm2_for_cuda {
    ($t:ident) => {
        fn nrm2(
            &self,
            x: &SharedTensor<$t>,
            result: &mut SharedTensor<$t>,
        ) -> Result<(), ::coaster::error::Error> {
            let n = x.desc().size() as i32;
            let x_mem = read!(x, self);
            let r_mem = write_only!(result, self);

            let ctx : &cublas::Context = self.framework().cublas();

            exec!(
                nrm2,
                (*ctx).nrm2(trans!(x_mem, $t), trans!(r_mem, $t), n, None)
            )
        }
    };
}

/// dot product for cuda
#[macro_export]
macro_rules! iblas_dot_for_cuda {
    ($t:ident) => {
        fn dot(
            &self,
            x: &SharedTensor<$t>,
            y: &SharedTensor<$t>,
            result: &mut SharedTensor<$t>,
        ) -> Result<(), ::coaster::error::Error> {
            let n = x.desc().size() as i32;
            let x_mem = read!(x, self);
            let y_mem = read!(y, self);
            let r_mem = write_only!(result, self);
            let ctx : &cublas::Context = self.framework().cublas();
            exec!(
                dot,
                (*ctx).dot(
                    trans!(x_mem, $t),
                    trans!(y_mem, $t),
                    trans!(r_mem, $t),
                    n,
                    None,
                    None
                )
            )
        }
    };
}

/// scalar mul for cuda
#[macro_export]
macro_rules! iblas_scal_for_cuda {
    ($t:ident) => {
        fn scal(
            &self,
            a: &SharedTensor<$t>,
            x: &mut SharedTensor<$t>,
        ) -> Result<(), ::coaster::error::Error> {
            let n = x.desc().size() as i32;
            let a_mem = read!(a, self);
            let x_mem = read_write!(x, self);
            let ctx : &cublas::Context = self.framework().cublas();

            exec!(
                scal,
                (*ctx).scal(trans!(a_mem, $t), trans!(x_mem, $t), n, None)
            )
        }
    };
}

/// swap matrices for cuda
#[macro_export]
macro_rules! iblas_swap_for_cuda {
    ($t:ident) => {
        fn swap(
            &self,
            x: &mut SharedTensor<$t>,
            y: &mut SharedTensor<$t>,
        ) -> Result<(), ::coaster::error::Error> {
            let n = x.desc().size() as i32;
            let x_mem = read_write!(x, self);
            let y_mem = read_write!(y, self);
            let ctx : &cublas::Context = self.framework().cublas();

            exec!(
                swap,
                (*ctx).swap(trans!(x_mem, $t), trans!(y_mem, $t), n, None, None)
            )
        }
    };
}

/// gbmv for cuda
#[macro_export]
macro_rules! iblas_gbmv_for_cuda {
    ($t:ident) => {
        fn gbmv(
            &self,
            _alpha: &SharedTensor<$t>,
            _at: Transpose,
            _a: &SharedTensor<$t>,
            _kl: &SharedTensor<u32>,
            _ku: &SharedTensor<u32>,
            _x: &SharedTensor<$t>,
            _beta: &SharedTensor<$t>,
            _c: &mut SharedTensor<$t>,
        ) -> Result<(), ::coaster::error::Error> {
            unimplemented!();
        }
    };
}

/// CUDA GEMM (Matrix-Matrix Multiplication)
///
/// Uses CUBLAS_V2 and SGEMM internally. CUBLAS excepts data to be provided in Column-Major spec,
/// although the rest of Juice uses Row-Major to store data. This conversion is 'ignored' by
/// substituting Matrix A for Matrix B and vice-versa in the multiplication. No additional
/// complexities should be expected from this.
#[macro_export]
macro_rules! iblas_gemm_for_cuda {
    ($t:ident) => {
        fn gemm(
            &self,
            alpha: &SharedTensor<$t>,
            at: Transpose,
            a: &SharedTensor<$t>,
            bt: Transpose,
            b: &SharedTensor<$t>,
            beta: &SharedTensor<$t>,
            c: &mut SharedTensor<$t>,
        ) -> Result<(), ::coaster::error::Error> {
            // Quick Guide to BLAS Notation;
            // Matrix A is of form m x k
            // Matrix B is of form k x n
            // Matrix C is of form m x n

            // m, n, and k are all to be taken as Op(m), Op(n), and Op(k), indicating their post
            // transposition values.

            // Leading Dimension is always equal or greater to the initial dimension of the matrix,
            // i.e. leading dimension of Matrix A is >= m, and normally m. This is not their post
            // transposition form, but their initial form in memory.

            let c_desc = c.desc().clone();
            let alpha_mem = read!(alpha, self);
            let beta_mem = read!(beta, self);
            let a_mem = read!(a, self);
            let b_mem = read!(b, self);
            let c_mem = write_only!(c, self);

            let a_0 = a.desc()[0] as i32;
            let a_1 = a.desc().iter().skip(1).fold(1, |prod, i| prod * i) as i32;
            let b_0 = b.desc()[0] as i32;
            let b_1 = b.desc().iter().skip(1).fold(1, |prod, i| prod * i) as i32;
            let c_1 = c_desc.iter().skip(1).fold(1, |prod, i| prod * i) as i32;
            let n = match bt {
                Transpose::NoTrans => b_1,
                _ => b_0,
            };
            let (m, k) = match at {
                Transpose::NoTrans => (a_0, a_1),
                _ => (a_1, a_0),
            };
            let lda = a_1;
            let ldb = b_1;
            let ldc = c_1;

            let ctx : &cublas::Context = self.framework().cublas();

            exec!(
                gemm,
                (*ctx).gemm(
                    ::cublas::api::Operation::from(bt),
                    ::cublas::api::Operation::from(at),
                    n, // n and m are switched for row-major support.
                    m,
                    k,
                    trans!(alpha_mem, $t),
                    trans!(b_mem, $t), // Memory for a & b are switched for row-major support.
                    ldb, // Leading dim of b and a are switched for row-major support.
                    trans!(a_mem, $t),
                    lda,
                    trans!(beta_mem, $t),
                    trans!(c_mem, $t),
                    ldc
                )
            )
        }

        fn gemm_batched(
            &self,
            alpha: &SharedTensor<$t>,
            at: Transpose,
            a: &SharedTensor<$t>,
            bt: Transpose,
            b: &SharedTensor<$t>,
            beta: &SharedTensor<$t>,
            c: &mut SharedTensor<$t>,
            batch_count: usize
        ) -> Result<(), ::coaster::error::Error> {
            let c_desc = c.desc().clone();
            let alpha_mem = read!(alpha, self);
            let beta_mem = read!(beta, self);
            let a_mem = read!(a, self);
            let b_mem = read!(b, self);
            let c_mem = write_only!(c, self);

            assert_eq!(a.desc().len(), 3);
            assert_eq!(b.desc().len(), 3);

            let a_batch = a.desc()[0];
            let b_batch = b.desc()[1];
            let c_batch = c_desc[2];

            let a_stride = a.desc().iter().skip(1).fold(1, |prod, i| prod * i) as i32;
            let b_stride = b.desc().iter().skip(1).fold(1, |prod, i| prod * i) as i32;
            let c_stride = c_desc.clone().iter().skip(1).fold(1, |prod, i| prod * i) as i32;

            let a_rows = a.desc()[1] as i32;
            let a_cols = a.desc()[2] as i32;

            let b_rows = b.desc()[1] as i32;
            let b_cols = b.desc()[2] as i32;

            let c_rows = c_desc[1] as i32;
            let c_cols = c_desc[2] as i32;

            let n = match bt {
                Transpose::NoTrans => b_cols,
                _ => b_rows,
            };
            let (m, k) = match at {
                Transpose::NoTrans => (a_rows, a_cols),
                _ => (a_cols, a_rows),
            };

            // Assumption of Row Major Input, independent of Transposition Ops
            let lda = a_cols;
            let ldb = b_cols;
            let ldc = c_cols;

            let ctx: &cublas::Context = self.framework().cublas();

            exec!(
                gemm_batched,
                (*ctx).gemm_batched(
                    ::cublas::api::Operation::from(bt),
                    ::cublas::api::Operation::from(at),
                    n, // n and m are switched for row-major support.
                    m,
                    k,
                    trans!(alpha_mem, $t),
                    trans!(b_mem, $t), // Memory for a & b are switched for row-major support.
                    ldb, // Leading dim of b and a are switched for row-major support.
                    b_stride as i64,
                    trans!(a_mem, $t),
                    lda,
                    a_stride as i64,
                    trans!(beta_mem, $t),
                    trans!(c_mem, $t),
                    ldc,
                    c_stride as i64,
                    batch_count as i32
                )
            )
        }
    };
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use coaster::frameworks::cuda::get_cuda_backend as cuda_backend;
    use coaster::*;
    use coaster::{SharedTensor, Backend, Native};
    use coaster::frameworks::native::get_native_backend;
    use coaster::plugin::numeric_helpers::NumCast;
    use coaster::plugin::numeric_helpers::cast;

    use std::rc::Rc;
    use crate::plugin::Gemm;
    use crate::transpose::Transpose;
    use coaster::prelude::FlatBox;

    pub fn write_to_memory_offset<T: NumCast + ::std::marker::Copy>(mem: &mut FlatBox, data: &[T], offset: usize) {
        let mem_buffer = mem.as_mut_slice::<f32>();
        for (index, datum) in data.iter().enumerate() {
            mem_buffer[index + offset] = cast(*datum).unwrap();
        }
    }

    pub fn write_to_memory<T: NumCast + ::std::marker::Copy>(mem: &mut FlatBox, data: &[T]) {
        write_to_memory_offset(mem, data, 0);
    }

    fn write_to_tensor(data: Vec<f32>, dims: Vec<usize>) -> SharedTensor<f32> {
        let native = get_native_backend();
        let mut tensor : SharedTensor<f32> = SharedTensor::new(&dims);
        write_to_memory(&mut tensor.write_only(native.device()).unwrap(), &data);
        tensor
    }

    fn read_from_tensor(tensor: SharedTensor<f32>) -> Vec<f32> {
        let native = get_native_backend();
        tensor.read(native.device())
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    }

    fn sample_rhs() -> SharedTensor<f32> {
        // Imitating an embedding layer
        let dims : Vec<usize> = vec![3,4];
        let data : Vec<f32> = vec![
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0
        ];
        write_to_tensor(data, dims)
    }

    fn batch_sample_rhs() -> SharedTensor<f32> {
        // Imitating an embedding layer
        let dims : Vec<usize> = vec![2, 3,4];
        let data : Vec<f32> = vec![
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0,

            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0
        ];
        write_to_tensor(data, dims)
    }

    fn sample_lhs() -> SharedTensor<f32> {
        // Imitating an Output Gradient Layer
        let dims : Vec<usize> = vec![2,4];
        let data : Vec<f32> = vec![
            1.0, 3.0, 5.0, 7.0,
            2.0, 4.0, 6.0, 8.0
        ];
        write_to_tensor(data, dims)
    }

    fn batch_sample_lhs() -> SharedTensor<f32> {
        // Imitating an Output Gradient Layer
        let dims : Vec<usize> = vec![2,2,4];
        let data : Vec<f32> = vec![
            1.0, 3.0, 5.0, 7.0,
            2.0, 4.0, 6.0, 8.0,

            1.0, 3.0, 5.0, 7.0,
            2.0, 4.0, 6.0, 8.0
        ];
        write_to_tensor(data, dims)
    }

    fn sample_output() -> Vec<f32> {
        vec![
            3.0, 7.0, 0.0,
            4.0, 8.0, 0.0
        ]
    }

    fn batch_sample_output() -> Vec<f32> {
        vec![
            3.0, 7.0, 0.0,
            4.0, 8.0, 0.0,

            3.0, 7.0, 0.0,
            4.0, 8.0, 0.0
        ]
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gemm() {
        let backend = Box::new(Rc::new(cuda_backend()));
        let mut test_output = SharedTensor::new(&[2,3]);
        backend.gemm(
            &write_to_tensor(vec![1.0], vec![1]),
            Transpose::NoTrans,
            &sample_lhs(),
            Transpose::Trans,
            &sample_rhs(),
            &write_to_tensor(vec![0.0], vec![1]),
            &mut test_output,
        );
        assert_eq!(read_from_tensor(test_output), sample_output());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gemm_batched() {
        let backend = Box::new(Rc::new(cuda_backend()));
        let mut test_output = SharedTensor::new(&[2,2,3]);
        backend.gemm_batched(
            &write_to_tensor(vec![1.0], vec![1]),
            Transpose::NoTrans,
            &batch_sample_lhs(),
            Transpose::Trans,
            &batch_sample_rhs(),
            &write_to_tensor(vec![0.0], vec![1]),
            &mut test_output,
            2
        );
        assert_eq!(read_from_tensor(test_output), batch_sample_output());
    }
}