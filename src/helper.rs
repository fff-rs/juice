//! Provides macros for convenient implementation of NN operations.

/// Returns cuDNN ready memory pointer from a SharedTensor.
pub unsafe fn receive_memory_ptr<T>(x: &::co::tensor::SharedTensor<T>, device: &::co::device::DeviceType) -> Result<*const ::libc::c_void, ::co::plugin::Error> {
    Ok(::std::mem::transmute::<u64, *const ::libc::c_void>(
        try!(
            try!(
                x.get(device).ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to resolve memory."))
            ).as_cuda().ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to receive CUDA memory."))
        ).id_c()
    ))
}

/// Returns mutable cuDNN ready memory pointer from a SharedTensor.
pub unsafe fn receive_memory_ptr_mut<T>(x: &::co::tensor::SharedTensor<T>, device: &::co::device::DeviceType) -> Result<*mut ::libc::c_void, ::co::plugin::Error> {
    Ok(::std::mem::transmute::<u64, *mut ::libc::c_void>(
        try!(
            try!(
                x.get(device).ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to resolve memory."))
            ).as_cuda().ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to receive CUDA memory."))
        ).id_c()
    ))
}

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

            Ok(try!(match self.binary().cudnn().sigmoid_forward(
                &try!(x.get_cudnn_desc()), // src_desc
                try!(unsafe { ::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(result.get_cudnn_desc()), // dest_desc
                try!(unsafe { ::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
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

            Ok(try!(match self.binary().cudnn().sigmoid_forward(
                &try!(x.get_cudnn_desc()), // src_desc
                try!(unsafe { ::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(result.get_cudnn_desc()), // dest_desc
                try!(unsafe { ::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
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

            Ok(try!(match self.binary().cudnn().sigmoid_backward(
                &try!(x.get_cudnn_desc()), // src_desc
                try!(unsafe { ::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(x_diff.get_cudnn_desc()), // src_diff_desc
                try!(unsafe { ::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
                &try!(result.get_cudnn_desc()), // dest_desc
                try!(unsafe { ::helper::receive_memory_ptr(result, self.device()) }), // dest_data
                &try!(result_diff.get_cudnn_desc()), // dest_diff_desc
                try!(unsafe { ::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
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

            Ok(try!(match self.binary().cudnn().sigmoid_backward(
                &try!(x.get_cudnn_desc()), // src_desc
                try!(unsafe { ::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(x_diff.get_cudnn_desc()), // src_diff_desc
                try!(unsafe { ::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
                &try!(result.get_cudnn_desc()), // dest_desc
                try!(unsafe { ::helper::receive_memory_ptr(result, self.device()) }), // dest_data
                &try!(result_diff.get_cudnn_desc()), // dest_diff_desc
                try!(unsafe { ::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
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

            Ok(try!(match self.binary().cudnn().relu_forward(
                &try!(x.get_cudnn_desc()), // src_desc
                try!(unsafe { ::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(result.get_cudnn_desc()), // dest_desc
                try!(unsafe { ::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
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

            Ok(try!(match self.binary().cudnn().relu_forward(
                &try!(x.get_cudnn_desc()), // src_desc
                try!(unsafe { ::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(result.get_cudnn_desc()), // dest_desc
                try!(unsafe { ::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
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

            Ok(try!(match self.binary().cudnn().relu_backward(
                &try!(x.get_cudnn_desc()), // src_desc
                try!(unsafe { ::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(x_diff.get_cudnn_desc()), // src_diff_desc
                try!(unsafe { ::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
                &try!(result.get_cudnn_desc()), // dest_desc
                try!(unsafe { ::helper::receive_memory_ptr(result, self.device()) }), // dest_data
                &try!(result_diff.get_cudnn_desc()), // dest_diff_desc
                try!(unsafe { ::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
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

            Ok(try!(match self.binary().cudnn().relu_backward(
                &try!(x.get_cudnn_desc()), // src_desc
                try!(unsafe { ::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(x_diff.get_cudnn_desc()), // src_diff_desc
                try!(unsafe { ::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
                &try!(result.get_cudnn_desc()), // dest_desc
                try!(unsafe { ::helper::receive_memory_ptr(result, self.device()) }), // dest_data
                &try!(result_diff.get_cudnn_desc()), // dest_diff_desc
                try!(unsafe { ::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
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

            Ok(try!(match self.binary().cudnn().tanh_forward(
                &try!(x.get_cudnn_desc()), // src_desc
                try!(unsafe { ::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(result.get_cudnn_desc()), // dest_desc
                try!(unsafe { ::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
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

            Ok(try!(match self.binary().cudnn().tanh_forward(
                &try!(x.get_cudnn_desc()), // src_desc
                try!(unsafe { ::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(result.get_cudnn_desc()), // dest_desc
                try!(unsafe { ::helper::receive_memory_ptr_mut(result, self.device()) }), // dest_data
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

            Ok(try!(match self.binary().cudnn().tanh_backward(
                &try!(x.get_cudnn_desc()), // src_desc
                try!(unsafe { ::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(x_diff.get_cudnn_desc()), // src_diff_desc
                try!(unsafe { ::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
                &try!(result.get_cudnn_desc()), // dest_desc
                try!(unsafe { ::helper::receive_memory_ptr(result, self.device()) }), // dest_data
                &try!(result_diff.get_cudnn_desc()), // dest_diff_desc
                try!(unsafe { ::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
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

            Ok(try!(match self.binary().cudnn().tanh_backward(
                &try!(x.get_cudnn_desc()), // src_desc
                try!(unsafe { ::helper::receive_memory_ptr(x, self.device()) }), //src_data
                &try!(x_diff.get_cudnn_desc()), // src_diff_desc
                try!(unsafe { ::helper::receive_memory_ptr(x_diff, self.device()) }), //src_diff_data
                &try!(result.get_cudnn_desc()), // dest_desc
                try!(unsafe { ::helper::receive_memory_ptr(result, self.device()) }), // dest_data
                &try!(result_diff.get_cudnn_desc()), // dest_diff_desc
                try!(unsafe { ::helper::receive_memory_ptr_mut(result_diff, self.device()) }), // dest_diff_data
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
