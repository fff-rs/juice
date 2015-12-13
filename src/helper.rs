//! Provides macros for convenient implementation of NN operations.

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
            let src_desc = try!(x.get_cudnn_desc());
            let src_data = try!(try!(x.get(self.device()).ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`")))
            .as_cuda().ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to receive native memory for `x`.")))
            .id_c();
            let dest_desc = try!(result.get_cudnn_desc());
            let dest_data = try!(try!(result.get_mut(self.device()).ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `result`")))
            .as_cuda().ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to receive native memory for `result`.")))
            .id_c();

            Ok(try!(match self.binary().cudnn().sigmoid_forward(
                &src_desc, unsafe { ::std::mem::transmute::<u64, *const ::libc::c_void>(src_data) },
                &dest_desc, unsafe { ::std::mem::transmute::<u64, *mut ::libc::c_void>(dest_data) },
                ::cudnn::ScalParams::<$t>::default()
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
            let src_desc = try!(x.get_cudnn_desc());
            let src_data = try!(try!(x.get(self.device()).ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`")))
            .as_cuda().ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to receive native memory for `x`.")))
            .id_c();
            let dest_desc = try!(result.get_cudnn_desc());
            let dest_data = try!(try!(result.get_mut(self.device()).ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `result`")))
            .as_cuda().ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to receive native memory for `result`.")))
            .id_c();

            Ok(try!(match self.binary().cudnn().sigmoid_forward(
                &src_desc, unsafe { ::std::mem::transmute::<u64, *const ::libc::c_void>(src_data) },
                &dest_desc, unsafe { ::std::mem::transmute::<u64, *mut ::libc::c_void>(dest_data) },
                ::cudnn::ScalParams::<$t>::default()
            ) {
                Ok(_) => Ok(()),
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation Sigmoid Forward."))
                }
            }))
        }

        fn sigmoid_diff(
            &self,
            x: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result.add_device(self.device()) { _ => () }
            let src_desc = try!(x.get_cudnn_desc());
            let src_data = try!(try!(x.get(self.device()).ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`")))
            .as_cuda().ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to receive native memory for `x`.")))
            .id_c();
            let dest_desc = try!(result.get_cudnn_desc());
            let dest_data = try!(try!(result.get_mut(self.device()).ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `result`")))
            .as_cuda().ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to receive native memory for `result`.")))
            .id_c();

            Ok(try!(match self.binary().cudnn().sigmoid_forward(
                &src_desc, unsafe { ::std::mem::transmute::<u64, *const ::libc::c_void>(src_data) },
                &dest_desc, unsafe { ::std::mem::transmute::<u64, *mut ::libc::c_void>(dest_data) },
                ::cudnn::ScalParams::<$t>::default()
            ) {
                Ok(_) => Ok(()),
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation Sigmoid Forward."))
                }
            }))
        }

        fn sigmoid_diff_plain(
            &self,
            x: &::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>
        ) -> Result<(), ::co::error::Error> {
            let src_desc = try!(x.get_cudnn_desc());
            let src_data = try!(try!(x.get(self.device()).ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`")))
            .as_cuda().ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to receive native memory for `x`.")))
            .id_c();
            let dest_desc = try!(result.get_cudnn_desc());
            let dest_data = try!(try!(result.get_mut(self.device()).ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `result`")))
            .as_cuda().ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to receive native memory for `result`.")))
            .id_c();

            Ok(try!(match self.binary().cudnn().sigmoid_forward(
                &src_desc, unsafe { ::std::mem::transmute::<u64, *const ::libc::c_void>(src_data) },
                &dest_desc, unsafe { ::std::mem::transmute::<u64, *mut ::libc::c_void>(dest_data) },
                ::cudnn::ScalParams::<$t>::default()
            ) {
                Ok(_) => Ok(()),
                Err(err) => {
                    println!("{:?}", err);
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation Sigmoid Forward."))
                }
            }))
        }
    )
}
