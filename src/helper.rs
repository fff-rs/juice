//! Provides macros for convenient implementation of NN operations.

#[macro_export]
macro_rules! impl_ops_sigmoid_for {
    ($t:ident, $b:ty) => (
        fn sigmoid(&self,
            x: &mut ::collenchyma::shared_memory::SharedMemory<$t, ::collenchyma::shared_memory::TensorR1>,
            result: &mut ::collenchyma::shared_memory::SharedMemory<$t, ::collenchyma::shared_memory::TensorR0>
        ) -> Result<(), ::collenchyma::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result.add_device(self.device()) { _ => () }
            Ok(try!(
                <$b as IOperationSigmoid<$t>>::compute(&self,
                    try!(x.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                    try!(result.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `result`"))),
                )
            ))
        }

        fn sigmoid_plain(&self,
            x: &mut ::collenchyma::shared_memory::SharedMemory<$t, ::collenchyma::shared_memory::TensorR1>,
            result: &mut ::collenchyma::shared_memory::SharedMemory<$t, ::collenchyma::shared_memory::TensorR0>
        ) -> Result<(), ::collenchyma::error::Error> {
            Ok(try!(
                <$b as IOperationSigmoid<$t>>::compute(&self,
                    try!(x.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                    try!(result.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `result`"))),
                )
            ))
        }
    );
}
