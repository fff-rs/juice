//! Provides useful macros for easier NN implementation for CUDA/cuDNN.

macro_rules! read {
    ($x:ident, $slf:ident) => (
        try!($x.read($slf.device())).as_cuda()
            .expect("Broken invariant: not a CUDA memory")
    )
}

macro_rules! read_write {
    ($x:ident, $slf:ident) => (
        try!($x.read_write($slf.device())).as_cuda()
            .expect("Broken invariant: not a CUDA memory")
    )
}

macro_rules! write_only {
    ($x:ident, $slf:ident) => (
        try!($x.write_only($slf.device())).as_cuda()
            .expect("Broken invariant: not a CUDA memory")
    )
}

// trans! cannot be inlined into macros above, because `$mem` would become
// intermidiate variable and `*mut $t` will outlive it.
macro_rules! trans {
    ($mem:ident) => (
        unsafe { ::std::mem::transmute::<u64, *const ::libc::c_void>(*$mem.id_c()) }
    )
}

macro_rules! trans_mut {
    ($mem:ident) => (
        unsafe { ::std::mem::transmute::<u64, *mut ::libc::c_void>(*$mem.id_c()) }
    )
}

macro_rules! exec {
    ($name:ident, $f:expr) => ({
        let res = $f;
        // FIXME: can we properly include original error?
        // Use String instead of &str or trait objects?
        res.map_err(|_| PluginError::Operation(
            stringify!(Unable to execute CUDA cuDNN $name)).into())
    })
}


#[macro_export]
macro_rules! impl_oconf_for_cc(($($t: ident), +) => (
    $(
        impl<'a> NNOperationConfig<$t> for utils::ConvolutionConfig { }
    )+
));

#[macro_export]
macro_rules! impl_oconf_for_clrn(($($t: ident), +) => (
    $(
        impl NNOperationConfig<$t> for utils::NormalizationConfig { }
    )+
));

#[macro_export]
macro_rules! impl_oconf_for_pooling(($($t: ident), +) => (
    $(
        impl NNOperationConfig<$t> for utils::PoolingConfig { }
    )+
));


// Implementation of Sigmoid, Relu, Tanh is mostly the same, excluding
// trait and function names. And it's quite big, so I think not repeating
// it here 3 times is worth another level of indirection.
// Since concat_idents!() is not stable, this macro has a lot of arguments.
#[macro_export]
macro_rules! impl_activation_ops {
    ($t:ty, $b:ty,
     $plugin_name:ident, $plugin_pointwise_name:ident,
     $fwd_cuda:ident, $bkw_cuda:ident,
     $fwd_name:ident, $bkw_name:ident,
     $fwd_pointwise_name:ident, $bkw_pointwise_name:ident) => (

        impl ::plugin::$plugin_name<$t> for $b {
            fn $fwd_name(&self, x: &SharedTensor<$t>, result: &mut SharedTensor<$t>)
                         -> Result<(), CoError> {
                let r_desc = try!(result.cudnn_tensor_desc_flat());
                let x_mem = read!(x, self);
                let r_mem = write_only!(result, self);

                exec!($fwd_name, CUDNN.$fwd_cuda(
                    &try!(x.cudnn_tensor_desc_flat()),
                    trans!(x_mem),
                    &r_desc,
                    trans_mut!(r_mem),
                    ScalParams::<$t>::default()))

            }

            fn $bkw_name(&self,
                         x: &SharedTensor<$t>,
                         x_diff: &SharedTensor<$t>,
                         result: &SharedTensor<$t>,
                         result_diff: &mut SharedTensor<$t>)
                         -> Result<(), CoError> {
                let dr_desc = try!(result_diff.cudnn_tensor_desc_flat());
                let x_mem = read!(x, self);
                let dx_mem = read!(x_diff, self);
                let r_mem = read!(result, self);
                let dr_mem = write_only!(result_diff, self);

                exec!($bkw_name, CUDNN.$bkw_cuda(
                    &try!(x.cudnn_tensor_desc_flat()),
                    trans!(x_mem),
                    &try!(x_diff.cudnn_tensor_desc_flat()),
                    trans!(dx_mem),
                    &try!(result.cudnn_tensor_desc_flat()),
                    trans!(r_mem),
                    &dr_desc,
                    trans_mut!(dr_mem),
                    ScalParams::<$t>::default()))
            }
        }

        impl ::plugin::$plugin_pointwise_name<$t> for $b {
            fn $fwd_pointwise_name(&self, x: &mut SharedTensor<$t>)
                                   -> Result<(), CoError> {
                let x_desc = try!(x.cudnn_tensor_desc_flat());
                let x_mem = read_write!(x, self);
                exec!($fwd_pointwise_name, CUDNN.$fwd_cuda(
                    &x_desc,
                    trans!(x_mem),
                    &x_desc,
                    trans_mut!(x_mem),
                    ScalParams::<$t>::default()))
            }

            fn $bkw_pointwise_name(&self, x: &SharedTensor<$t>,
                                   x_diff: &mut SharedTensor<$t>)
                                   -> Result<(), CoError> {
                let x_desc = try!(x.cudnn_tensor_desc_flat());
                let dx_desc = try!(x_diff.cudnn_tensor_desc_flat());
                let x_mem = read!(x, self);
                let dx_mem = read_write!(x_diff, self);
                exec!($bkw_pointwise_name, CUDNN.$bkw_cuda(
                    &x_desc, trans!(x_mem),
                    &dx_desc, trans!(dx_mem),
                    &x_desc, trans!(x_mem),
                    &dx_desc, trans_mut!(dx_mem),
                    ScalParams::<$t>::default()))
            }
        }
    )
}

macro_rules! impl_ops_sigmoid_for {
    ($t:ty, $b:ty) => (
        impl_activation_ops!(
            $t, $b,
            Sigmoid, SigmoidPointwise,
            sigmoid_forward, sigmoid_backward,
            sigmoid, sigmoid_grad,
            sigmoid_pointwise, sigmoid_pointwise_grad);
    )
}

macro_rules! impl_ops_relu_for {
    ($t:ty, $b:ty) => (
        impl_activation_ops!(
            $t, $b,
            Relu, ReluPointwise,
            relu_forward, relu_backward,
            relu, relu_grad,
            relu_pointwise, relu_pointwise_grad);
    )
}

macro_rules! impl_ops_tanh_for {
    ($t:ty, $b:ty) => (
        impl_activation_ops!(
            $t, $b,
            Tanh, TanhPointwise,
            tanh_forward, tanh_backward,
            tanh, tanh_grad,
            tanh_pointwise, tanh_pointwise_grad);
    )
}



#[macro_export]
macro_rules! impl_ops_convolution_for {
    ($t:ty, $b:ty) => (
        fn convolution(
            &self,
            filter: &SharedTensor<$t>,
            x: &SharedTensor<$t>,
            result: &mut SharedTensor<$t>,
            workspace: &mut SharedTensor<u8>,
            config: &Self::CC) -> Result<(), CoError> {

            let r_desc = try!(result.cudnn_tensor_desc());
            let f_mem = read!(filter, self);
            let x_mem = read!(x, self);
            let r_mem = write_only!(result, self);
            let w_mem = write_only!(workspace, self);

            exec!(convolution, CUDNN.convolution_forward::<$t>(
                config,
                trans_mut!(w_mem),
                trans!(f_mem),
                &try!(x.cudnn_tensor_desc()), // src_desc
                trans!(x_mem),
                &r_desc,
                trans_mut!(r_mem),
                ScalParams::default()))
        }

        #[allow(unused_variables)]
        fn convolution_grad_filter(
            &self,
            src_data: &SharedTensor<$t>,
            dest_diff: &SharedTensor<$t>,
            filter_diff: &mut SharedTensor<$t>,
            workspace: &mut SharedTensor<u8>,
            config: &Self::CC) -> Result<(), CoError> {

            let s_mem = read!(src_data, self);
            let dd_mem = read!(dest_diff, self);
            let df_mem = write_only!(filter_diff, self);
            let w_mem = write_only!(workspace, self);
            exec!(convolution_grad_filter, CUDNN.convolution_backward_filter(
                config,
                trans_mut!(w_mem),
                &try!(src_data.cudnn_tensor_desc()),
                trans!(s_mem),
                &try!(dest_diff.cudnn_tensor_desc()),
                trans!(dd_mem),
                trans_mut!(df_mem),
                ScalParams::<$t>::default()))
        }

        #[allow(unused_variables)]
        fn convolution_grad_data(
            &self,
            filter: &SharedTensor<$t>,
            x_diff: &SharedTensor<$t>,
            result_diff: &mut SharedTensor<$t>,
            workspace: &mut SharedTensor<u8>,
            config: &Self::CC) -> Result<(), CoError> {

            let dr_desc = try!(result_diff.cudnn_tensor_desc());
            let f_mem = read!(filter, self);
            let dx_mem = read!(x_diff, self);
            let dr_mem = write_only!(result_diff, self);
            let w_mem = write_only!(workspace, self);
            exec!(convolution_grad_data, CUDNN.convolution_backward_data(
                config,
                trans_mut!(w_mem),
                trans!(f_mem),
                &try!(x_diff.cudnn_tensor_desc()),
                trans!(dx_mem),
                &dr_desc,
                trans_mut!(dr_mem),
                ScalParams::<$t>::default()))
        }
    )
}

#[macro_export]
macro_rules! impl_ops_softmax_for {
    ($t:ty, $b:ty) => (
        impl ::plugin::Softmax<$t> for $b {
            fn softmax(&self, x: &SharedTensor<$t>, result: &mut SharedTensor<$t>)
                       -> Result<(), CoError> {
                let r_desc = try!(result.cudnn_tensor_desc_softmax());
                let x_mem = read!(x, self);
                let r_mem = write_only!(result, self);
                exec!(softmax, CUDNN.softmax_forward(
                    &try!(x.cudnn_tensor_desc_softmax()),
                    trans!(x_mem),
                    &r_desc,
                    trans_mut!(r_mem),
                    ScalParams::<$t>::default()))
            }

            fn softmax_grad(
                &self,
                x: &SharedTensor<$t>,
                x_diff: &SharedTensor<$t>,
                result_diff: &mut SharedTensor<$t>) -> Result<(), CoError> {

                let dr_desc = try!(result_diff.cudnn_tensor_desc_softmax());
                let x_mem = read!(x, self);
                let dx_mem = read!(x_diff, self);
                let dr_mem = write_only!(result_diff, self);

                exec!(softmax_backward, CUDNN.softmax_backward(
                    &try!(x.cudnn_tensor_desc_softmax()),
                    trans!(x_mem),
                    &try!(x_diff.cudnn_tensor_desc_softmax()),
                    trans!(dx_mem),
                    &dr_desc,
                    trans_mut!(dr_mem),
                    ScalParams::<$t>::default()))
            }
        }
    )
}

#[macro_export]
macro_rules! impl_ops_log_softmax_for {
    ($t:ty, $b:ty) => (
        impl ::plugin::LogSoftmax<$t> for $b {
            fn log_softmax(&self, x: &SharedTensor<$t>, result: &mut SharedTensor<$t>)
                       -> Result<(), CoError> {
                let r_desc = try!(result.cudnn_tensor_desc_softmax());
                let x_mem = read!(x, self);
                let r_mem = write_only!(result, self);
                exec!(log_softmax, CUDNN.log_softmax_forward(
                    &try!(x.cudnn_tensor_desc_softmax()),
                    trans!(x_mem),
                    &r_desc,
                    trans_mut!(r_mem),
                    ScalParams::<$t>::default()))
            }

            fn log_softmax_grad(
                &self,
                x: &SharedTensor<$t>,
                x_diff: &SharedTensor<$t>,
                result_diff: &mut SharedTensor<$t>) -> Result<(), CoError> {

                let dr_desc = try!(result_diff.cudnn_tensor_desc_softmax());
                let x_mem = read!(x, self);
                let dx_mem = read!(x_diff, self);
                let dr_mem = write_only!(result_diff, self);

                exec!(log_softmax_backward, CUDNN.log_softmax_backward(
                    &try!(x.cudnn_tensor_desc_softmax()),
                    trans!(x_mem),
                    &try!(x_diff.cudnn_tensor_desc_softmax()),
                    trans!(dx_mem),
                    &dr_desc,
                    trans_mut!(dr_mem),
                    ScalParams::<$t>::default()))
            }
        }
    )
}

#[macro_export]
macro_rules! impl_ops_lrn_for {
    ($t:ty, $b:ty) => (
        impl ::plugin::LRN<$t> for $b {
            fn new_lrn_config(&self, n: u32, alpha: f64, beta: f64, k: f64)
                              -> Result<Self::CLRN, CoError> {
                Ok(CUDNN.init_normalization(n, alpha, beta, k)
                .map_err(|_| PluginError::Operation("Failed to create cuDNN PoolingConfig"))?)
            }

            fn lrn(&self, x: &SharedTensor<$t>, result: &mut SharedTensor<$t>,
                   config: &Self::CLRN) -> Result<(), CoError> {
                let r_desc = try!(result.cudnn_tensor_desc());
                let x_mem = read!(x, self);
                let r_mem = write_only!(result, self);
                exec!(lrn_forward, CUDNN.lrn_forward(
                    config,
                    &try!(x.cudnn_tensor_desc()),
                    trans!(x_mem),
                    &r_desc,
                    trans_mut!(r_mem),
                    ScalParams::<$t>::default()))
            }

            #[allow(unused_variables)]
            fn lrn_grad(
                &self,
                x: &SharedTensor<$t>,
                x_diff: &SharedTensor<$t>,
                result: &SharedTensor<$t>,
                result_diff: &mut SharedTensor<$t>,
                config: &Self::CLRN) -> Result<(), CoError> {

                let dr_desc = try!(result_diff.cudnn_tensor_desc());
                let x_mem = read!(x, self);
                let dx_mem = read!(x_diff, self);
                let r_mem = read!(result, self);
                let dr_mem = write_only!(result_diff, self);
                exec!(lrn_backward, CUDNN.lrn_backward(
                    config,
                    &try!(x.cudnn_tensor_desc()),
                    trans!(x_mem),
                    &try!(x_diff.cudnn_tensor_desc()),
                    trans!(dx_mem),
                    &try!(result.cudnn_tensor_desc()),
                    trans!(r_mem),
                    &dr_desc,
                    trans_mut!(dr_mem),
                    ScalParams::<$t>::default()))
            }
        }
    )
}

#[macro_export]
macro_rules! impl_ops_pooling_for {
    ($t:ty, $b:ty) => (
        impl ::plugin::Pooling<$t> for $b {
            fn new_pooling_config(&self, window: &[i32], padding: &[i32],
                                  stride: &[i32]) -> Result<Self::CPOOL, CoError> {
                let pooling_avg = PoolingDescriptor::new(
                    cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                    window, padding, stride)
                    .map_err(|_| PluginError::Operation("Failed to create cuDNN avg PoolingConfig"))?;

                let pooling_max = PoolingDescriptor::new(
                    cudnnPoolingMode_t::CUDNN_POOLING_MAX,
                    window, padding, stride)
                    .map_err(|_| PluginError::Operation("Failed to create cuDNN max PoolingConfig"))?;
                Ok(utils::PoolingConfig::new(pooling_avg, pooling_max))
            }

            fn pooling_max(&self, x: &SharedTensor<$t>, result: &mut SharedTensor<$t>,
                           config: &Self::CPOOL) -> Result<(), CoError> {
                let r_desc = try!(result.cudnn_tensor_desc());
                let x_mem = read!(x, self);
                let r_mem = write_only!(result, self);
                exec!(pooling_max_forward, CUDNN.pooling_max_forward(
                    config,
                    &try!(x.cudnn_tensor_desc()),
                    trans!(x_mem),
                    &r_desc,
                    trans_mut!(r_mem),
                    ScalParams::<$t>::default()))
            }

            #[allow(unused_variables)]
            fn pooling_max_grad(
                &self,
                x: &SharedTensor<$t>,
                x_diff: &SharedTensor<$t>,
                result: &SharedTensor<$t>,
                result_diff: &mut SharedTensor<$t>,
                config: &Self::CPOOL) -> Result<(), CoError> {

                let dr_desc = try!(result_diff.cudnn_tensor_desc());
                let x_mem = read!(x, self);
                let dx_mem = read!(x_diff, self);
                let r_mem = read!(result, self);
                let dr_mem = write_only!(result_diff, self);
                exec!(pooling_max_backward, CUDNN.pooling_max_backward(
                    config,
                    &try!(x.cudnn_tensor_desc()),
                    trans!(x_mem),
                    &try!(x_diff.cudnn_tensor_desc()),
                    trans!(dx_mem),
                    &try!(result.cudnn_tensor_desc()),
                    trans!(r_mem),
                    &dr_desc,
                    trans_mut!(dr_mem),
                    ScalParams::<$t>::default()))
            }

            fn pooling_avg(&self, x: &SharedTensor<$t>, result: &mut SharedTensor<$t>,
                           config: &Self::CPOOL) -> Result<(), CoError> {
                let r_desc = try!(result.cudnn_tensor_desc());
                let x_mem = read!(x, self);
                let r_mem = write_only!(result, self);
                exec!(pooling_avg_forward, CUDNN.pooling_avg_forward(
                    config,
                    &try!(x.cudnn_tensor_desc()),
                    trans!(x_mem),
                    &r_desc,
                    trans_mut!(r_mem),
                    ScalParams::<$t>::default()))
            }

            #[allow(unused_variables)]
            fn pooling_avg_grad(
                &self,
                x: &SharedTensor<$t>,
                x_diff: &SharedTensor<$t>,
                result: &SharedTensor<$t>,
                result_diff: &mut SharedTensor<$t>,
                config: &Self::CPOOL) -> Result<(), CoError> {

                let dr_desc = try!(result_diff.cudnn_tensor_desc());
                let x_mem = read!(x, self);
                let dx_mem = read!(x_diff, self);
                let r_mem = read!(result, self);
                let dr_mem = write_only!(result_diff, self);
                exec!(pooling_avg_backward, CUDNN.pooling_avg_backward(
                    config,
                    &try!(x.cudnn_tensor_desc()),
                    trans!(x_mem),
                    &try!(x_diff.cudnn_tensor_desc()),
                    trans!(dx_mem),
                    &try!(result.cudnn_tensor_desc()),
                    trans!(r_mem),
                    &dr_desc,
                    trans_mut!(dr_mem),
                    ScalParams::<$t>::default()))
            }
        }
    )
}
