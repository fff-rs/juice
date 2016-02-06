#[macro_export]
macro_rules! iblas_asum_for_cuda {
    ($t:ident) => (
        fn asum(&self,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>,
            result: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result.add_device(self.device()) { _ => try!(result.sync(self.device())) }
            self.asum_plain(x, result)
        }

        fn asum_plain(&self,
            x: &::collenchyma::tensor::SharedTensor<$t>,
            result: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            let x_get = try!(x.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`")));
            let r_get = try!(result.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `result`")));
            let x_mem = try!(x_get.as_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive CUDA memory for `x`.")));
            let r_mem = try!(r_get.as_mut_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive CUDA memory for `result`.")));
            unsafe {
                let res = CONTEXT.asum(::std::mem::transmute::<u64, *mut $t>(*x_mem.id_c()),
                                 ::std::mem::transmute::<u64, *mut $t>(*r_mem.id_c()),
                                 x.desc().size() as i32,
                                 None);
                if res.is_ok() {
                    Ok(())
                } else {
                    Err(::collenchyma::error::Error::Plugin(::collenchyma::plugin::Error::Operation("Unable to execute operation asum.")))
                }
            }
        }
    );
}

#[macro_export]
macro_rules! iblas_axpy_for_cuda {
    ($t:ident) => (
        fn axpy(&self,
            a: &mut ::collenchyma::tensor::SharedTensor<$t>,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>,
            y: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            match a.add_device(self.device()) { _ => try!(a.sync(self.device())) }
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match y.add_device(self.device()) { _ => try!(y.sync(self.device())) }
            self.axpy_plain(a, x, y)
        }

        fn axpy_plain(&self,
            a: &::collenchyma::tensor::SharedTensor<$t>,
            x: &::collenchyma::tensor::SharedTensor<$t>,
            y: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            let n = x.desc().size() as i32;
            let a_get = try!(a.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `a`")));
            let x_get = try!(x.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`")));
            let y_get = try!(y.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `y`")));
            let a_mem = try!(a_get.as_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive CUDA memory for `a`.")));
            let x_mem = try!(x_get.as_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive CUDA memory for `x`.")));
            let y_mem = try!(y_get.as_mut_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive CUDA memory for `y`.")));
            unsafe {
                let res = CONTEXT.axpy(::std::mem::transmute::<u64, *mut $t>(*a_mem.id_c()),
                                       ::std::mem::transmute::<u64, *mut $t>(*x_mem.id_c()),
                                       ::std::mem::transmute::<u64, *mut $t>(*y_mem.id_c()),
                                       n,
                                       None,
                                       None);
                if res.is_ok() {
                    Ok(())
                } else {
                    Err(::collenchyma::error::Error::Plugin(::collenchyma::plugin::Error::Operation("Unable to execute operation axpy.")))
                }
            }
        }
    );
}

#[macro_export]
macro_rules! iblas_copy_for_cuda {
    ($t:ident) => (
        fn copy(&self,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>,
            y: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match y.add_device(self.device()) { _ => try!(y.sync(self.device())) }
            self.copy_plain(x, y)
        }

        fn copy_plain(&self,
            x: &::collenchyma::tensor::SharedTensor<$t>,
            y: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            let n = x.desc().size() as i32;
            let x_get = try!(x.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`")));
            let y_get = try!(y.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `y`")));
            let x_mem = try!(x_get.as_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive CUDA memory for `x`.")));
            let y_mem = try!(y_get.as_mut_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive CUDA memory for `y`.")));
            unsafe {
                let res = CONTEXT.copy(::std::mem::transmute::<u64, *mut $t>(*x_mem.id_c()),
                                       ::std::mem::transmute::<u64, *mut $t>(*y_mem.id_c()),
                                       n,
                                       None,
                                       None);
                if res.is_ok() {
                    Ok(())
                } else {
                    Err(::collenchyma::error::Error::Plugin(::collenchyma::plugin::Error::Operation("Unable to execute operation copy.")))
                }
            }
        }
    );
}

#[macro_export]
macro_rules! iblas_dot_for_cuda {
    ($t:ident) => (
        fn dot(&self,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>,
            y: &mut ::collenchyma::tensor::SharedTensor<$t>,
            result: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match y.add_device(self.device()) { _ => try!(y.sync(self.device())) }
            match result.add_device(self.device()) { _ => try!(result.sync(self.device())) }
            self.dot_plain(x, y, result)
        }

        fn dot_plain(&self,
            x: &::collenchyma::tensor::SharedTensor<$t>,
            y: &::collenchyma::tensor::SharedTensor<$t>,
            result: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            let n = x.desc().size() as i32;
            let x_get = try!(x.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`")));
            let y_get = try!(y.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `y`")));
            let r_get = try!(result.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `result`")));
            let x_mem = try!(x_get.as_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive CUDA memory for `x`.")));
            let y_mem = try!(y_get.as_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive CUDA memory for `y`.")));
            let r_mem = try!(r_get.as_mut_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive CUDA memory for `result`.")));
            unsafe {
                let res = CONTEXT.dot( ::std::mem::transmute::<u64, *mut $t>(*x_mem.id_c()),
                                       ::std::mem::transmute::<u64, *mut $t>(*y_mem.id_c()),
                                       ::std::mem::transmute::<u64, *mut $t>(*r_mem.id_c()),
                                       n,
                                       None,
                                       None);
                if res.is_ok() {
                    Ok(())
                } else {
                    println!("{:?}", res);
                    Err(::collenchyma::error::Error::Plugin(::collenchyma::plugin::Error::Operation("Unable to execute operation dot.")))
                }
            }
        }
    );
}

#[macro_export]
macro_rules! iblas_nrm2_for_cuda {
    ($t:ident) => (
        fn nrm2(&self,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>,
            result: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result.add_device(self.device()) { _ => try!(result.sync(self.device())) }
            self.nrm2_plain(x, result)
        }

        fn nrm2_plain(&self,
            x: &::collenchyma::tensor::SharedTensor<$t>,
            result: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            let n = x.desc().size() as i32;
            let x_get = try!(x.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`")));
            let r_get = try!(result.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `result`")));
            let x_mem = try!(x_get.as_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive CUDA memory for `x`.")));
            let r_mem = try!(r_get.as_mut_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive CUDA memory for `result`.")));
            unsafe {
                let res = CONTEXT.nrm2(::std::mem::transmute::<u64, *mut $t>(*x_mem.id_c()),
                                       ::std::mem::transmute::<u64, *mut $t>(*r_mem.id_c()),
                                       n,
                                       None);
                if res.is_ok() {
                    Ok(())
                } else {
                    Err(::collenchyma::error::Error::Plugin(::collenchyma::plugin::Error::Operation("Unable to execute operation nrm2.")))
                }
            }
        }
    );
}

#[macro_export]
macro_rules! iblas_scal_for_cuda {
    ($t:ident) => (
        fn scal(&self,
            a: &mut ::collenchyma::tensor::SharedTensor<$t>,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            match a.add_device(self.device()) { _ => try!(a.sync(self.device())) }
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            self.scal_plain(a, x)
        }

        fn scal_plain(&self,
            a: &::collenchyma::tensor::SharedTensor<$t>,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            let n = x.desc().size() as i32;
            let a_get = try!(a.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `a`")));
            let x_get = try!(x.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`")));
            let a_mem = try!(a_get.as_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive CUDA memory for `a`.")));
            let x_mem = try!(x_get.as_mut_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive CUDA memory for `x`.")));
            unsafe {
                let res = CONTEXT.scal(::std::mem::transmute::<u64, *mut $t>(*a_mem.id_c()),
                                       ::std::mem::transmute::<u64, *mut $t>(*x_mem.id_c()),
                                       n,
                                       None);
                if res.is_ok() {
                    Ok(())
                } else {
                    Err(::collenchyma::error::Error::Plugin(::collenchyma::plugin::Error::Operation("Unable to execute operation scal.")))
                }
            }
        }
    );
}

#[macro_export]
macro_rules! iblas_swap_for_cuda {
    ($t:ident) => (
        fn swap(&self,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>,
            y: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match y.add_device(self.device()) { _ => try!(y.sync(self.device())) }
            self.swap_plain(x, y)
        }

        fn swap_plain(&self,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>,
            y: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            let n = x.desc().size() as i32;
            let x_get = try!(x.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`")));
            let y_get = try!(y.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `y`")));
            let x_mem = try!(x_get.as_mut_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive CUDA memory for `x`.")));
            let y_mem = try!(y_get.as_mut_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive CUDA memory for `y`.")));
            unsafe {
                let res = CONTEXT.swap(::std::mem::transmute::<u64, *mut $t>(*x_mem.id_c()),
                                       ::std::mem::transmute::<u64, *mut $t>(*y_mem.id_c()),
                                       n,
                                       None,
                                       None);
                if res.is_ok() {
                    Ok(())
                } else {
                    Err(::collenchyma::error::Error::Plugin(::collenchyma::plugin::Error::Operation("Unable to execute operation swap.")))
                }
            }
        }
    );
}

#[macro_export]
macro_rules! iblas_gemm_for_cuda {
    ($t:ident) => (
        fn gemm(&self,
            alpha: &mut ::collenchyma::tensor::SharedTensor<$t>,
            at: ::transpose::Transpose,
            a: &mut ::collenchyma::tensor::SharedTensor<$t>,
            bt: ::transpose::Transpose,
            b: &mut ::collenchyma::tensor::SharedTensor<$t>,
            beta: &mut ::collenchyma::tensor::SharedTensor<$t>,
            c: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            match alpha.add_device(self.device()) { _ => try!(alpha.sync(self.device())) }
            match a.add_device(self.device()) { _ => try!(a.sync(self.device())) }
            match beta.add_device(self.device()) { _ => try!(beta.sync(self.device())) }
            match b.add_device(self.device()) { _ => try!(b.sync(self.device())) }
            match c.add_device(self.device()) { _ => try!(c.sync(self.device())) }
            self.gemm_plain(alpha, at, a, bt, b, beta, c)
        }

        fn gemm_plain(&self,
            alpha: &::collenchyma::tensor::SharedTensor<$t>,
            at: ::transpose::Transpose,
            a: &::collenchyma::tensor::SharedTensor<$t>,
            bt: ::transpose::Transpose,
            b: &::collenchyma::tensor::SharedTensor<$t>,
            beta: &::collenchyma::tensor::SharedTensor<$t>,
            c: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            let c_desc = c.desc().clone();
            let alpha_get = try!(alpha.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `alpha`")));
            let alpha_mem = try!(alpha_get.as_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive CUDA memory for `alpha`.")));
            let a_get = try!(a.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `a`")));
            let a_mem = try!(a_get.as_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive CUDA memory for `a`.")));
            let b_get = try!(b.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `b`")));
            let b_mem = try!(b_get.as_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive CUDA memory for `b`.")));
            let beta_get = try!(beta.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `beta`")));
            let beta_mem = try!(beta_get.as_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive CUDA memory for `beta`.")));
            let c_get = try!(c.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `c`")));
            let c_mem = try!(c_get.as_mut_cuda().ok_or(PluginError::MissingMemoryForDevice("Unable to receive CUDA memory for `c`.")));
            unsafe {
                let a_0 = a.desc()[0] as i32;
                let a_1 = a.desc().iter().skip(1).fold(1, |prod, i| prod * i) as i32;
                let b_0 = b.desc()[0] as i32;
                let b_1 = b.desc().iter().skip(1).fold(1, |prod, i| prod * i) as i32;
                let c_1 = c_desc.iter().skip(1).fold(1, |prod, i| prod * i) as i32;
                let n = match bt {
                    ::transpose::Transpose::NoTrans => b_1,
                    _ => b_0
                };
                let (m, k) = match at {
                    ::transpose::Transpose::NoTrans => (a_0, a_1),
                    _ => (a_1, a_0)
                };
                let lda = a_1;
                let ldb = b_1;
                let ldc = c_1;
                // println!("A desc {:?}", a.desc());
                // println!("B desc {:?}", b.desc());
                // println!("C desc {:?}", c_desc);
                // println!("lda {:?}", lda);
                // println!("ldb {:?}", ldb);
                // println!("ldc {:?}", ldc);
                // println!("M {:?}", m);
                // println!("N {:?}", n);
                // println!("K {:?}", k);
                let res = CONTEXT.gemm(::cublas::api::Operation::from(bt),
                                       ::cublas::api::Operation::from(at),
                                       n, m, k,
                                       ::std::mem::transmute::<u64, *mut $t>(*alpha_mem.id_c()),
                                       ::std::mem::transmute::<u64, *mut $t>(*b_mem.id_c()), // matrix a and b are switched to make it work with row-major memory layout.
                                       ldb,
                                       ::std::mem::transmute::<u64, *mut $t>(*a_mem.id_c()),
                                       lda,
                                       ::std::mem::transmute::<u64, *mut $t>(*beta_mem.id_c()),
                                       ::std::mem::transmute::<u64, *mut $t>(*c_mem.id_c()),
                                       ldc);
                if res.is_ok() {
                    Ok(())
                } else {
                    Err(::collenchyma::error::Error::Plugin(::collenchyma::plugin::Error::Operation("Unable to execute operation gemm.")))
                }
            }
        }
    );
}
