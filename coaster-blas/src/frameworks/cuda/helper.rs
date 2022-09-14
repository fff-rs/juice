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
            log::debug!(r#"Unable to execute operation: {}
{:?}"#, stringify!($name), e);
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

            let ctx: &cublas::Context = self.framework().cublas();
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
            assert_eq!(x.desc().size(), y.desc().size());
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

            let ctx: &cublas::Context = self.framework().cublas();

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
            let ctx: &cublas::Context = self.framework().cublas();
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
            let ctx: &cublas::Context = self.framework().cublas();

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
            let ctx: &cublas::Context = self.framework().cublas();

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

/// gemm for cuda
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
            use Transpose as T;

            // Determine the dimensions of all the matrices.
            // We always treat the first dimension as the number of rows and all
            // the subsequent dimensions combined as the "columns".
            let a_0 = a.desc()[0] as i32;
            let a_1 = a.desc().iter().skip(1).fold(1, |prod, i| prod * i) as i32;
            let b_0 = b.desc()[0] as i32;
            let b_1 = b.desc().iter().skip(1).fold(1, |prod, i| prod * i) as i32;
            let c_0 = c.desc()[0] as i32;
            let c_1 = c.desc().iter().skip(1).fold(1, |prod, i| prod * i) as i32;

            let (m, n, k) = match (at, bt) {
                (T::NoTrans, T::NoTrans) => {
                    assert_eq!(a_1, b_0);
                    (a_0, b_1, a_1)
                }
                (T::NoTrans, T::Trans | T::ConjTrans) => {
                    assert_eq!(a_1, b_1);
                    (a_0, b_0, a_1)
                }
                (T::Trans | T::ConjTrans, T::NoTrans) => {
                    assert_eq!(a_0, b_0);
                    (a_1, b_1, a_0)
                }
                (T::Trans | T::ConjTrans, T::Trans | T::ConjTrans) => {
                    assert_eq!(a_0, b_1);
                    (a_1, b_0, a_0)
                }
            };

            // Verify that C dimensions match.
            assert_eq!(c_0, m);
            assert_eq!(c_1, n);

            let lda = a_1;
            let ldb = b_1;
            let ldc = c_1;

            let ctx: &cublas::Context = self.framework().cublas();

            let alpha_mem = read!(alpha, self);
            let beta_mem = read!(beta, self);
            let a_mem = read!(a, self);
            let b_mem = read!(b, self);
            let c_mem = write_only!(c, self);

            // cuBLAS uses column-major matrix format, while SharedTensor is row-major.
            // To compute AxB = C, we instead compute BᵀxAᵀ = Cᵀ and treat the transposed
            // column-major matrix as a normal (non-transposed) row-major one.
            exec!(
                gemm,
                (*ctx).gemm(
                    ::cublas::api::Operation::from(bt),
                    ::cublas::api::Operation::from(at),
                    n,
                    m,
                    k,
                    trans!(alpha_mem, $t),
                    trans!(b_mem, $t),
                    ldb,
                    trans!(a_mem, $t),
                    lda,
                    trans!(beta_mem, $t),
                    trans!(c_mem, $t),
                    ldc
                )
            )
        }
    };
}
