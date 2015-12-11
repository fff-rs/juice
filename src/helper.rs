//! Provides macros for convenient implementation of BLAS operations.

#[macro_export]
macro_rules! iblas_asum_for {
    ($t:ident, $b:ty) => (
        fn asum(&self,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>,
            result: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result.add_device(self.device()) { _ => () }
            Ok(try!(
                <$b as IOperationAsum<$t>>::compute(&self,
                    try!(x.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                    try!(result.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `result`"))),
                )
            ))
        }

        fn asum_plain(&self,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>,
            result: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            Ok(try!(
                <$b as IOperationAsum<$t>>::compute(&self,
                    try!(x.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                    try!(result.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `result`"))),
                )
            ))
        }
    );
}

#[macro_export]
macro_rules! iblas_axpy_for {
    ($t:ident, $b:ty) => (
        fn axpy(&self,
            a: &mut ::collenchyma::tensor::SharedTensor<$t>,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>,
            y: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            match a.add_device(self.device()) { _ => try!(a.sync(self.device())) }
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match y.add_device(self.device()) { _ => try!(y.sync(self.device())) }
            Ok(try!(
                <$b as IOperationAxpy<$t>>::compute(&self,
                    try!(a.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `a`"))),
                    try!(x.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                    try!(y.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `y`"))),
                )
            ))
        }

        fn axpy_plain(&self,
            a: &mut ::collenchyma::tensor::SharedTensor<$t>,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>,
            y: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            Ok(try!(
                <$b as IOperationAxpy<$t>>::compute(&self,
                    try!(a.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `a`"))),
                    try!(x.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                    try!(y.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `y`"))),
                )
            ))
        }
    );
}

#[macro_export]
macro_rules! iblas_copy_for {
    ($t:ident, $b:ty) => (
        fn copy(&self,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>,
            y: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match y.add_device(self.device()) { _ => () }
            Ok(try!(
                <$b as IOperationCopy<$t>>::compute(&self,
                    try!(x.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                    try!(y.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `y`"))),
                )
            ))
        }

        fn copy_plain(&self,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>,
            y: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            Ok(try!(
                <$b as IOperationCopy<$t>>::compute(&self,
                    try!(x.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                    try!(y.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `y`"))),
                )
            ))
        }
    );
}

#[macro_export]
macro_rules! iblas_dot_for {
    ($t:ident, $b:ty) => (
        fn dot(&self,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>,
            y: &mut ::collenchyma::tensor::SharedTensor<$t>,
            result: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match y.add_device(self.device()) { _ => try!(y.sync(self.device())) }
            match result.add_device(self.device()) { _ => () }
            Ok(try!(
                <$b as IOperationDot<$t>>::compute(&self,
                    try!(x.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                    try!(y.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `y`"))),
                    try!(result.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `result`")))
                )
            ))
        }

        fn dot_plain(&self,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>,
            y: &mut ::collenchyma::tensor::SharedTensor<$t>,
            result: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            Ok(try!(
                <$b as IOperationDot<$t>>::compute(&self,
                    try!(x.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                    try!(y.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `y`"))),
                    try!(result.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `result`")))
                )
            ))
        }
    );
}

#[macro_export]
macro_rules! iblas_nrm2_for {
    ($t:ident, $b:ty) => (
        fn nrm2(&self,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>,
            result: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result.add_device(self.device()) { _ => () }
            Ok(try!(
                <$b as IOperationNrm2<$t>>::compute(&self,
                    try!(x.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                    try!(result.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `result`"))),
                )
            ))
        }

        fn nrm2_plain(&self,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>,
            result: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            Ok(try!(
                <$b as IOperationNrm2<$t>>::compute(&self,
                    try!(x.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                    try!(result.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `result`"))),
                )
            ))
        }
    );
}

#[macro_export]
macro_rules! iblas_scale_for {
    ($t:ident, $b:ty) => (
        fn scale(&self,
            a: &mut ::collenchyma::tensor::SharedTensor<$t>,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            match a.add_device(self.device()) { _ => try!(a.sync(self.device())) }
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            Ok(try!(
                <$b as IOperationScale<$t>>::compute(&self,
                    try!(a.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `a`"))),
                    try!(x.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                )
            ))
        }

        fn scale_plain(&self,
            a: &mut ::collenchyma::tensor::SharedTensor<$t>,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            Ok(try!(
                <$b as IOperationScale<$t>>::compute(&self,
                    try!(a.get(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `a`"))),
                    try!(x.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                )
            ))
        }
    );
}

#[macro_export]
macro_rules! iblas_swap_for {
    ($t:ident, $b:ty) => (
        fn swap(&self,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>,
            y: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match y.add_device(self.device()) { _ => try!(y.sync(self.device())) }
            Ok(try!(
                <$b as IOperationSwap<$t>>::compute(&self,
                    try!(x.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                    try!(y.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `y`"))),
                )
            ))
        }

        fn swap_plain(&self,
            x: &mut ::collenchyma::tensor::SharedTensor<$t>,
            y: &mut ::collenchyma::tensor::SharedTensor<$t>
        ) -> Result<(), ::collenchyma::error::Error> {
            Ok(try!(
                <$b as IOperationSwap<$t>>::compute(&self,
                    try!(x.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                    try!(y.get_mut(self.device()).ok_or(::collenchyma::plugin::Error::MissingMemoryForDevice("Unable to resolve memory for `y`"))),
                )
            ))
        }
    );
}
