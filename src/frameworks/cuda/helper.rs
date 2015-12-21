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
        fn sigmoid(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result.add_device(self.device()) { _ => () }
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(try!(match CUDNN.sigmoid_forward(
                &try!(x.cudnn_tensor_desc()), // src_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(result.cudnn_tensor_desc()), // dest_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
                scal_params
            ) {
                Ok(_) => Ok(()),
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation Sigmoid Forward."))
                }
            }))
        }

        fn sigmoid_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(try!(match CUDNN.sigmoid_forward(
                &try!(x.cudnn_tensor_desc()), // src_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(result.cudnn_tensor_desc()), // dest_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
                scal_params
            ) {
                Ok(_) => Ok(()),
                Err(err) => {
                    println!("{:?}", err);
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
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(try!(match CUDNN.sigmoid_backward(
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
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation Sigmoid Backward."))
                }
            }))
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
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation Sigmoid Backward."))
                }
            }))
        }
    )
}

#[macro_export]
macro_rules! impl_ops_relu_for {
    ($t:ident, $b:ty) => (
        fn relu(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result.add_device(self.device()) { _ => () }
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(try!(match CUDNN.relu_forward(
                &try!(x.cudnn_tensor_desc()), // src_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(result.cudnn_tensor_desc()), // dest_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
                scal_params
            ) {
                Ok(_) => Ok(()),
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation relu Forward."))
                }
            }))
        }

        fn relu_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(try!(match CUDNN.relu_forward(
                &try!(x.cudnn_tensor_desc()), // src_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(result.cudnn_tensor_desc()), // dest_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
                scal_params
            ) {
                Ok(_) => Ok(()),
                Err(err) => {
                    println!("{:?}", err);
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
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(try!(match CUDNN.relu_backward(
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
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation relu Backward."))
                }
            }))
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
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation relu Backward."))
                }
            }))
        }
    )
}

#[macro_export]
macro_rules! impl_ops_tanh_for {
    ($t:ident, $b:ty) => (
        fn tanh(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result.add_device(self.device()) { _ => () }
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(try!(match CUDNN.tanh_forward(
                &try!(x.cudnn_tensor_desc()), // src_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(result.cudnn_tensor_desc()), // dest_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
                scal_params
            ) {
                Ok(_) => Ok(()),
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation tanh Forward."))
                }
            }))
        }

        fn tanh_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(try!(match CUDNN.tanh_forward(
                &try!(x.cudnn_tensor_desc()), // src_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(result.cudnn_tensor_desc()), // dest_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
                scal_params
            ) {
                Ok(_) => Ok(()),
                Err(err) => {
                    println!("{:?}", err);
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
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(try!(match CUDNN.tanh_backward(
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
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation tanh Backward."))
                }
            }))
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
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation tanh Backward."))
                }
            }))
        }
    )
}

#[macro_export]
macro_rules! impl_ops_convolution_for {
    ($t:ident, $b:ty) => (
        fn convolution(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CC //::frameworks::cuda::CC
        ) -> Result<(), ::co::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result.add_device(self.device()) { _ => () }
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(try!(match CUDNN.convolution_forward(
                config,
                &try!(x.cudnn_tensor_desc()), // src_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(result.cudnn_tensor_desc()), // dest_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
                scal_params
            ) {
                Ok(_) => Ok(()),
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation convolution Forward."))
                }
            }))
        }

        fn convolution_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CC
        ) -> Result<(), ::co::error::Error> {
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(try!(match CUDNN.convolution_forward(
                config,
                &try!(x.cudnn_tensor_desc()), // src_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(result.cudnn_tensor_desc()), // dest_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
                scal_params
            ) {
                Ok(_) => Ok(()),
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation convolution Forward."))
                }
            }))
        }

        #[allow(unused_variables)]
        fn convolution_grad(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            x_diff: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CC
        ) -> Result<(), ::co::error::Error> {
            match x_diff.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result_diff.add_device(self.device()) { _ => () }
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(try!(match CUDNN.convolution_backward(
                config,
                &try!(x_diff.cudnn_tensor_desc()), // src_diff_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
                &try!(result_diff.cudnn_tensor_desc()), // dest_diff_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
                scal_params
            ) {
                Ok(_) => Ok(()),
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation convolution Backward."))
                }
            }))
        }

        #[allow(unused_variables)]
        fn convolution_grad_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            x_diff: &::co::tensor::SharedTensor<$t>,
            result: &::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>,
            config: &Self::CC
        ) -> Result<(), ::co::error::Error> {
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(try!(match CUDNN.convolution_backward(
                config,
                &try!(x_diff.cudnn_tensor_desc()), // src_diff_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
                &try!(result_diff.cudnn_tensor_desc()), // dest_diff_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
                scal_params
            ) {
                Ok(_) => Ok(()),
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation convolution Backward."))
                }
            }))
        }
    )
}

#[macro_export]
macro_rules! impl_ops_softmax_for {
    ($t:ident, $b:ty) => (
        fn softmax(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result.add_device(self.device()) { _ => () }
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(try!(match CUDNN.softmax_forward(
                &try!(x.cudnn_tensor_desc()), // src_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(result.cudnn_tensor_desc()), // dest_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
                scal_params
            ) {
                Ok(_) => Ok(()),
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation softmax Forward."))
                }
            }))
        }

        fn softmax_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(try!(match CUDNN.softmax_forward(
                &try!(x.cudnn_tensor_desc()), // src_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(result.cudnn_tensor_desc()), // dest_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
                scal_params
            ) {
                Ok(_) => Ok(()),
                Err(err) => {
                    println!("{:?}", err);
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
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(try!(match CUDNN.softmax_backward(
                &try!(x.cudnn_tensor_desc()), // src_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(x_diff.cudnn_tensor_desc()), // src_diff_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
                &try!(result_diff.cudnn_tensor_desc()), // dest_diff_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
                scal_params
            ) {
                Ok(_) => Ok(()),
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation softmax Backward."))
                }
            }))
        }

        fn softmax_grad_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            x_diff: &::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(try!(match CUDNN.softmax_backward(
                &try!(x.cudnn_tensor_desc()), // src_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(x_diff.cudnn_tensor_desc()), // src_diff_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
                &try!(result_diff.cudnn_tensor_desc()), // dest_diff_desc
                try!(unsafe { ::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
                scal_params
            ) {
                Ok(_) => Ok(()),
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation softmax Backward."))
                }
            }))
        }
    )
}

#[macro_export]
macro_rules! impl_ops_lrn_for {
    ($t:ident, $b:ty) => (
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
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation lrn Forward."))
                }
            }))
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
                Err(err) => {
                    println!("{:?}", err);
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
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation lrn Backward."))
                }
            }))
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
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation lrn Backward."))
                }
            }))
        }
    )
}

#[macro_export]
macro_rules! impl_ops_pooling_for {
    ($t:ident, $b:ty) => (
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
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation pooling Forward."))
                }
            }))
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
                Err(err) => {
                    println!("{:?}", err);
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
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation pooling Backward."))
                }
            }))
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
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation pooling Backward."))
                }
            }))
        }
    )
}
