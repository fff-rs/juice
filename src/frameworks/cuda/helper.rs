//! Provides useful macros for easier NN implementation for CUDA/cuDNN.

/// Returns cuDNN ready memory pointer from a SharedTensor.
pub unsafe fn receive_memory_ptr<T>(x: &::co::tensor::SharedTensor<T>, device: &::co::device::DeviceType) -> Result<*const ::libc::c_void, ::co::plugin::Error> {
    Ok(::std::mem::transmute::<u64, *const ::libc::c_void>(
        *try!(
            try!(
                x.get(device).ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to resolve memory."))
            ).as_cuda().ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to receive CUDA memory."))
        ).id_c()
    ))
}

/// Returns mutable cuDNN ready memory pointer from a SharedTensor.
pub unsafe fn receive_memory_ptr_mut<T>(x: &mut ::co::tensor::SharedTensor<T>, device: &::co::device::DeviceType) -> Result<*mut ::libc::c_void, ::co::plugin::Error> {
    Ok(::std::mem::transmute::<u64, *mut ::libc::c_void>(
        *try!(
            try!(
                x.get_mut(device).ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to resolve memory."))
            ).as_mut_cuda().ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to receive CUDA memory."))
        ).id_c()
    ))
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

#[macro_export]
macro_rules! impl_ops_sigmoid_for {
    ($t:ident, $b:ty) => (
        impl ::plugin::Sigmoid<$t> for $b {
            fn sigmoid(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match result.add_device(self.device()) { _ => () }

                self.sigmoid_plain(x, result)
            }

            fn sigmoid_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

                Ok(try!(match CUDNN.sigmoid_forward(
                    &try!(x.cudnn_tensor_desc_flat()), // src_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                    &try!(result.cudnn_tensor_desc_flat()), // dest_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation Sigmoid Forward."))
                    }
                }))
            }

            fn sigmoid_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match x_diff.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match result.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match result_diff.add_device(self.device()) { _ => () }

                self.sigmoid_grad_plain(x, x_diff, result, result_diff)
            }

            fn sigmoid_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &::co::tensor::SharedTensor<$t>,
                result: &::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

                Ok(try!(match CUDNN.sigmoid_backward(
                    &try!(x.cudnn_tensor_desc_flat()), // src_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                    &try!(x_diff.cudnn_tensor_desc_flat()), // src_diff_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
                    &try!(result.cudnn_tensor_desc_flat()), // dest_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(result, self.device()) }), // dest_data
                    &try!(result_diff.cudnn_tensor_desc_flat()), // dest_diff_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation Sigmoid Backward."))
                    }
                }))
            }
        }
    )
}

#[macro_export]
macro_rules! impl_ops_relu_for {
    ($t:ident, $b:ty) => (
        impl ::plugin::Relu<$t> for $b {
            fn relu(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match result.add_device(self.device()) { _ => () }

                self.relu_plain(x, result)
            }

            fn relu_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

                Ok(try!(match CUDNN.relu_forward(
                    &try!(x.cudnn_tensor_desc_flat()), // src_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                    &try!(result.cudnn_tensor_desc_flat()), // dest_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation relu Forward."))
                    }
                }))
            }

            fn relu_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match x_diff.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match result.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match result_diff.add_device(self.device()) { _ => () }

                self.relu_grad_plain(x, x_diff, result, result_diff)
            }

            fn relu_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &::co::tensor::SharedTensor<$t>,
                result: &::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

                Ok(try!(match CUDNN.relu_backward(
                    &try!(x.cudnn_tensor_desc_flat()), // src_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                    &try!(x_diff.cudnn_tensor_desc_flat()), // src_diff_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
                    &try!(result.cudnn_tensor_desc_flat()), // dest_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(result, self.device()) }), // dest_data
                    &try!(result_diff.cudnn_tensor_desc_flat()), // dest_diff_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation relu Backward."))
                    }
                }))
            }
        }
    )
}

#[macro_export]
macro_rules! impl_ops_tanh_for {
    ($t:ident, $b:ty) => (
        impl ::plugin::Tanh<$t> for $b {
            fn tanh(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match result.add_device(self.device()) { _ => () }

                self.tanh_plain(x, result)
            }

            fn tanh_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

                Ok(try!(match CUDNN.tanh_forward(
                    &try!(x.cudnn_tensor_desc_flat()), // src_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                    &try!(result.cudnn_tensor_desc_flat()), // dest_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation tanh Forward."))
                    }
                }))
            }

            fn tanh_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match x_diff.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match result.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match result_diff.add_device(self.device()) { _ => () }

                self.tanh_grad_plain(x, x_diff, result, result_diff)
            }

            fn tanh_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &::co::tensor::SharedTensor<$t>,
                result: &::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

                Ok(try!(match CUDNN.tanh_backward(
                    &try!(x.cudnn_tensor_desc_flat()), // src_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                    &try!(x_diff.cudnn_tensor_desc_flat()), // src_diff_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
                    &try!(result.cudnn_tensor_desc_flat()), // dest_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(result, self.device()) }), // dest_data
                    &try!(result_diff.cudnn_tensor_desc_flat()), // dest_diff_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation tanh Backward."))
                    }
                }))
            }
        }
    )
}

#[macro_export]
macro_rules! impl_ops_convolution_for {
    ($t:ty, $b:ty) => (
        fn convolution(
            &self,
            filter: &mut ::co::tensor::SharedTensor<$t>,
            x: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CC //::frameworks::cuda::CC
        ) -> Result<(), ::co::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result.add_device(self.device()) { _ => () }

            self.convolution_plain(filter, x, result, config)
        }

        fn convolution_plain(
            &self,
            filter: &::co::tensor::SharedTensor<$t>,
            x: &::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CC
        ) -> Result<(), ::co::error::Error> {
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(try!(match CUDNN.convolution_forward(
                config,
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(filter, self.device()) }),
                &try!(x.cudnn_tensor_desc()), // src_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(result.cudnn_tensor_desc()), // dest_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
                scal_params
            ) {
                Ok(_) => Ok(()),
                Err(_) => {
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation convolution Forward."))
                }
            }))
        }

        #[allow(unused_variables)]
        fn convolution_grad_filter(
            &self,
            src_data: &mut ::co::tensor::SharedTensor<$t>,
            dest_diff: &mut ::co::tensor::SharedTensor<$t>,
            filter_diff: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CC
        ) -> Result<(), ::co::error::Error> {
            match src_data.add_device(self.device()) { _ => try!(src_data.sync(self.device())) }
            match dest_diff.add_device(self.device()) { _ => try!(dest_diff.sync(self.device())) }
            match filter_diff.add_device(self.device()) { _ => try!(filter_diff.sync(self.device())) }

            self.convolution_grad_filter_plain(src_data, dest_diff, filter_diff, config)
        }

        #[allow(unused_variables)]
        fn convolution_grad_filter_plain(
            &self,
            src_data: &::co::tensor::SharedTensor<$t>,
            dest_diff: &::co::tensor::SharedTensor<$t>,
            filter_diff: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CC
        ) -> Result<(), ::co::error::Error> {
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(try!(match CUDNN.convolution_backward_filter(
                config,
                &try!(src_data.cudnn_tensor_desc()),
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(src_data, self.device()) }),
                &try!(dest_diff.cudnn_tensor_desc()),
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(dest_diff, self.device()) }),
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(filter_diff, self.device()) }),
                scal_params
            ) {
                Ok(_) => Ok(()),
                Err(_) => {
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation convolution Backward."))
                }
            }))
        }

        #[allow(unused_variables)]
        fn convolution_grad_data(
            &self,
            filter: &mut ::co::tensor::SharedTensor<$t>,
            x_diff: &mut ::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CC
        ) -> Result<(), ::co::error::Error> {
            match filter.add_device(self.device()) { _ => try!(filter.sync(self.device())) }
            match x_diff.add_device(self.device()) { _ => try!(x_diff.sync(self.device())) }
            match result_diff.add_device(self.device()) { _ => try!(result_diff.sync(self.device())) }

            self.convolution_grad_data_plain(filter, x_diff, result_diff, config)
        }

        #[allow(unused_variables)]
        fn convolution_grad_data_plain(
            &self,
            filter: &::co::tensor::SharedTensor<$t>,
            x_diff: &::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CC
        ) -> Result<(), ::co::error::Error> {
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(try!(match CUDNN.convolution_backward_data(
                config,
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(filter, self.device()) }),
                &try!(x_diff.cudnn_tensor_desc()),
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }),
                &try!(result_diff.cudnn_tensor_desc()),
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }),
                scal_params
            ) {
                Ok(_) => Ok(()),
                Err(_) => {
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation convolution Backward."))
                }
            }))
        }
    )
}

#[macro_export]
macro_rules! impl_ops_softmax_for {
    ($t:ident, $b:ty) => (
        impl ::plugin::Softmax<$t> for $b {
            fn softmax(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match result.add_device(self.device()) { _ => () }

                self.softmax_plain(x, result)
            }

            fn softmax_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

                Ok(try!(match CUDNN.softmax_forward(
                    &try!(x.cudnn_tensor_desc_softmax()), // src_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                    &try!(result.cudnn_tensor_desc_softmax()), // dest_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation softmax Forward."))
                    }
                }))
            }

            fn softmax_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match x_diff.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match result_diff.add_device(self.device()) { _ => () }

                self.softmax_grad_plain(x, x_diff, result_diff)
            }

            fn softmax_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

                Ok(try!(match CUDNN.softmax_backward(
                    &try!(x.cudnn_tensor_desc_softmax()), // src_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                    &try!(x_diff.cudnn_tensor_desc_softmax()), // src_diff_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
                    &try!(result_diff.cudnn_tensor_desc_softmax()), // dest_diff_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation softmax Backward."))
                    }
                }))
            }
        }
    )
}

#[macro_export]
macro_rules! impl_ops_lrn_for {
    ($t:ident, $b:ty) => (
        impl ::plugin::LRN<$t> for $b {
            fn new_lrn_config(
                &self,
                n: u32,
                alpha: f64,
                beta: f64,
                k: f64
            ) -> Result<Self::CLRN, ::co::error::Error> {
                Ok(CUDNN.init_normalization(n, alpha, beta, k).unwrap())
            }

            fn lrn(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CLRN //::frameworks::cuda::CC
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match result.add_device(self.device()) { _ => () }

                self.lrn_plain(x, result, config)
            }

            fn lrn_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CLRN
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

                Ok(try!(match CUDNN.lrn_forward(
                    config,
                    &try!(x.cudnn_tensor_desc()), // src_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                    &try!(result.cudnn_tensor_desc()), // dest_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation lrn Forward."))
                    }
                }))
            }

            #[allow(unused_variables)]
            fn lrn_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CLRN
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match x_diff.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match result.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match result_diff.add_device(self.device()) { _ => () }

                self.lrn_grad_plain(x, x_diff, result, result_diff, config)
            }

            #[allow(unused_variables)]
            fn lrn_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &::co::tensor::SharedTensor<$t>,
                result: &::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CLRN
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

                Ok(try!(match CUDNN.lrn_backward(
                    config,
                    &try!(x.cudnn_tensor_desc()), // src_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                    &try!(x_diff.cudnn_tensor_desc()), // src_diff_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
                    &try!(result.cudnn_tensor_desc()), // dest_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(result, self.device()) }), // dest_data
                    &try!(result_diff.cudnn_tensor_desc()), // dest_diff_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation lrn Backward."))
                    }
                }))
            }
        }
    )
}

#[macro_export]
macro_rules! impl_ops_pooling_for {
    ($t:ident, $b:ty) => (
        impl ::plugin::Pooling<$t> for $b {
            fn new_pooling_config(
                &self,
                window: &[i32],
                padding: &[i32],
                stride: &[i32],
            ) -> Result<Self::CPOOL, ::co::error::Error> {
                let pooling_avg = ::cudnn::PoolingDescriptor::new(::cudnn::cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, window, padding, stride).unwrap();
                let pooling_max = ::cudnn::PoolingDescriptor::new(::cudnn::cudnnPoolingMode_t::CUDNN_POOLING_MAX, window, padding, stride).unwrap();
                Ok(::cudnn::utils::PoolingConfig::new(pooling_avg, pooling_max))
            }

            fn pooling_max(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CPOOL
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match result.add_device(self.device()) { _ => () }

                self.pooling_max_plain(x, result, config)
            }

            fn pooling_max_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CPOOL
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

                Ok(try!(match CUDNN.pooling_max_forward(
                    config,
                    &try!(x.cudnn_tensor_desc()), // src_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                    &try!(result.cudnn_tensor_desc()), // dest_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation pooling Forward."))
                    }
                }))
            }

            #[allow(unused_variables)]
            fn pooling_max_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CPOOL
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match x_diff.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match result.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match result_diff.add_device(self.device()) { _ => () }

                self.pooling_max_grad_plain(x, x_diff, result, result_diff, config)
            }

            #[allow(unused_variables)]
            fn pooling_max_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &::co::tensor::SharedTensor<$t>,
                result: &::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CPOOL
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

                Ok(try!(match CUDNN.pooling_max_backward(
                    config,
                    &try!(x.cudnn_tensor_desc()), // src_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                    &try!(x_diff.cudnn_tensor_desc()), // src_diff_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
                    &try!(result.cudnn_tensor_desc()), // dest_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(result, self.device()) }), // dest_data
                    &try!(result_diff.cudnn_tensor_desc()), // dest_diff_desc
                    try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation pooling Backward."))
                    }
                }))
            }
        }
    )
}
