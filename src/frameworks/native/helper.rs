//! Provides useful macros for easier NN implementation for native.

use co::plugin::numeric_helpers::Float;
use co::memory::MemoryType;

#[derive(Debug, Copy, Clone)]
#[allow(missing_docs)]
pub struct ConvolutionConfig;
#[derive(Debug, Copy, Clone)]
#[allow(missing_docs)]
pub struct NormalizationConfig;
#[derive(Debug, Copy, Clone)]
#[allow(missing_docs)]
pub struct PoolingConfig;

/// Just a helper function until SharedTensor has a nice interface for writing data
pub fn write_to_memory<T: Iterator>(mem: &mut MemoryType, data: T)
where T::Item: Clone {
    match mem {
        &mut MemoryType::Native(ref mut mem) => {
            let mut mem_buffer = mem.as_mut_slice::<T::Item>();
            for (index, datum) in data.enumerate() {
                mem_buffer[index] = datum;
            }
        },
        #[cfg(any(feature = "opencl", feature = "cuda"))]
        _ => {}
    }
}

#[inline]
/// Computes the Sigmoid Function on the CPU
pub fn sigmoid<T: Float>(x: &T) -> T {
    let x : T = x.clone();
    (T::one()) / (T::one() + (-x).exp())
}

#[inline]
/// Computes the Sigmoid Gradient on the CPU
pub fn sigmoid_grad<T: Float>(x: &T, dx: &T) -> T {
    let x : T = x.clone();
    let dx : T = dx.clone();
    x * (T::one() -x) * dx
}

#[inline]
/// Computes the ReLU Function on the CPU
pub fn relu<T: Float>(x: &T) -> T {
    let x : T = x.clone();
    x.max(T::zero())
}

#[inline]
/// Computes the ReLU Gradient on the CPU
pub fn relu_grad<T: Float>(x: &T, dx: &T) -> T {
    let x : T = x.clone();
    let dx : T = dx.clone();
    if x > T::zero() {
        return dx
    }
    T::zero()
}

#[inline]
/// Computes the Tanh Function on the CPU
pub fn tanh<T: Float>(x: &T) -> T {
    let x : T = x.clone();
    x.tanh()
}

#[inline]
// d/dx tanh x = sech2 x = 1 + tanh2 x
/// Computes the Tanh Gradient on the CPU
pub fn tanh_grad<T: Float>(x: &T, dx: &T) -> T {
    let x : T = x.clone();
    let dx : T = dx.clone();
    (T::one() - x.powi(2)) * dx
}

macro_rules! impl_oconf_for_cc(($($t: ident), +) => (
    $(
        impl<'a> NNOperationConfig<$t> for ::frameworks::native::helper::ConvolutionConfig { }
    )+
));

macro_rules! impl_oconf_for_clrn(($($t: ident), +) => (
    $(
        impl NNOperationConfig<$t> for ::frameworks::native::helper::NormalizationConfig { }
    )+
));

macro_rules! impl_oconf_for_pooling(($($t: ident), +) => (
    $(
        impl NNOperationConfig<$t> for ::frameworks::native::helper::PoolingConfig { }
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
                if let Some(input) = x.get(self.device()).unwrap().as_native() {
                    let res = input.as_slice::<$t>().iter().map(::frameworks::native::helper::sigmoid);
                    ::frameworks::native::helper::write_to_memory(result.get_mut(self.device()).unwrap(), res);
                    return Ok(());
                }
                Err(Error::Plugin(PluginError::Operation("Unable to execute Native sigmoid Forward.")))
            }

            fn sigmoid_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match x_diff.add_device(self.device()) { _ => try!(x_diff.sync(self.device())) }
                match result.add_device(self.device()) { _ => try!(result.sync(self.device())) }
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
                if let Some(sig_data) = x.get(self.device()).unwrap().as_native() {
                    if let Some(sig_dx) = x_diff.get(self.device()).unwrap().as_native() {
                        let res = sig_data.as_slice::<$t>().iter()
                        .zip(sig_dx.as_slice::<$t>().iter())
                        .map(|(t, dt)| ::frameworks::native::helper::sigmoid_grad(t, dt));
                        ::frameworks::native::helper::write_to_memory(result_diff.get_mut(self.device()).unwrap(), res);
                        return Ok(());
                    }
                }
                Err(Error::Plugin(PluginError::Operation("Unable to execute Native sigmoid grad Forward.")))
            }
        }
    );
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
                if let Some(input) = x.get(self.device()).unwrap().as_native() {
                    let res = input.as_slice::<$t>().iter().map(::frameworks::native::helper::relu);
                    ::frameworks::native::helper::write_to_memory(result.get_mut(self.device()).unwrap(), res);
                    return Ok(());
                }
                Err(Error::Plugin(PluginError::Operation("Unable to execute Native ReLU Forward.")))
            }

            fn relu_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match x_diff.add_device(self.device()) { _ => try!(x_diff.sync(self.device())) }
                match result.add_device(self.device()) { _ => try!(result.sync(self.device())) }
                self.relu_grad_plain(x, x_diff, result, result_diff)
            }

            fn relu_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &::co::tensor::SharedTensor<$t>,
                result: &::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                if let Some(input) = x.get(self.device()).unwrap().as_native() {
                    if let Some(dx) = x_diff.get(self.device()).unwrap().as_native() {
                        let res = input.as_slice::<$t>().iter()
                        .zip(dx.as_slice::<$t>().iter())
                        .map(|(x, dx)| ::frameworks::native::helper::relu_grad(x, dx));
                        ::frameworks::native::helper::write_to_memory(result_diff.get_mut(self.device()).unwrap(), res);
                        return Ok(());
                    }
                }
                Err(Error::Plugin(PluginError::Operation("Unable to execute Native ReLU grad Forward.")))
            }
        }
    );
}

#[macro_export]
macro_rules! impl_ops_tanh_for {
    ($t:ident, $b:ty) => (
        impl ::plugin::Tanh<$t> for $b {
            #[inline]
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
                if let Some(input) = x.get(self.device()).unwrap().as_native() {
                    let res = input.as_slice::<$t>().iter().map(::frameworks::native::helper::tanh);
                    ::frameworks::native::helper::write_to_memory(result.get_mut(self.device()).unwrap(), res);
                    return Ok(());
                }
                Err(Error::Plugin(PluginError::Operation("Unable to execute Native tanh Forward.")))
            }

            fn tanh_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match x_diff.add_device(self.device()) { _ => try!(x_diff.sync(self.device())) }
                match result.add_device(self.device()) { _ => try!(result.sync(self.device())) }
                self.tanh_grad_plain(x, x_diff, result, result_diff)
            }

            fn tanh_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &::co::tensor::SharedTensor<$t>,
                result: &::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                if let Some(input) = x.get(self.device()).unwrap().as_native() {
                    if let Some(dx) = x_diff.get(self.device()).unwrap().as_native() {
                        let res = input.as_slice::<$t>().iter()
                        .zip(dx.as_slice::<$t>().iter())
                        .map(|(x, dx)| ::frameworks::native::helper::tanh_grad(x, dx));
                        ::frameworks::native::helper::write_to_memory(result_diff.get_mut(self.device()).unwrap(), res);
                        return Ok(());
                    }
                }
                Err(Error::Plugin(PluginError::Operation("Unable to execute Native tanh_grad Forward.")))
            }
        }
    );
}

#[macro_export]
macro_rules! impl_ops_convolution_for {
    ($t:ident, $b:ty) => (
        impl ::plugin::Convolution<$t> for $b {
            fn new_convolution_config(
                &self,
                src: &::co::tensor::SharedTensor<$t>,
                dest: &::co::tensor::SharedTensor<$t>,
                filter: &mut ::co::tensor::SharedTensor<$t>,
                stride: &[i32],
                zero_padding: &[i32]
            ) -> Result<Self::CC, ::co::error::Error> {
                unimplemented!();
                Ok(helper::ConvolutionConfig)
            }
            fn convolution(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CC
            ) -> Result<(), ::co::error::Error> {
                unimplemented!();
                Ok(())
            }

            fn convolution_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CC
            ) -> Result<(), ::co::error::Error> {
                unimplemented!();
                Ok(())
            }

            fn convolution_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CC
            ) -> Result<(), ::co::error::Error> {
                unimplemented!();
                Ok(())
            }

            fn convolution_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &::co::tensor::SharedTensor<$t>,
                result: &::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CC
            ) -> Result<(), ::co::error::Error> {
                unimplemented!();
                Ok(())
            }
        }
    );
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
                if let Some(input) = x.get(self.device()).unwrap().as_native() {
                    let mut exps = Vec::with_capacity(x.capacity());
                    let mut sum : $t = 0 as $t;
                    for exp in input.as_slice::<$t>().iter().map(|t|t.exp()) {
                        exps.push(exp);
                        sum += exp;
                    }
                    let res = exps.iter().map(|t| t / sum);
                    ::frameworks::native::helper::write_to_memory(result.get_mut(self.device()).unwrap(), res);
                    return Ok(());
                }
                Err(Error::Plugin(
                    PluginError::Operation("Unable to execute Native softmax Forward.")))
            }
            fn softmax_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
                match x_diff.add_device(self.device()) { _ => try!(x_diff.sync(self.device())) }
                match result_diff.add_device(self.device()) { _ => () }
                self.softmax_grad_plain(x, x_diff, result_diff)
            }
            fn softmax_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                if let Some(sig_data) = x.get(self.device()).unwrap().as_native() {
                    if let Some(sig_dx) = x_diff.get(self.device()).unwrap().as_native() {
                        let mut dot : $t = 0 as $t;
                        let sig_data_slice = sig_data.as_slice::<$t>();
                        let sig_dx_slice = sig_dx.as_slice::<$t>();
                        for (t, dt) in sig_data_slice.iter().zip(sig_dx_slice.iter()) {
                            dot += t * dt;
                        }
                        let res = sig_data_slice.iter()
                            .zip(sig_dx_slice.iter())
                            .map(|(t, dt)| t * (dt - dot));
                        ::frameworks::native::helper::write_to_memory(result_diff.get_mut(self.device()).unwrap(), res);
                        return Ok(());
                    }
                }
                Err(Error::Plugin(
                        PluginError::Operation("Unable to execute Native softmax Backward.")))

            }
        }
    );
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
                unimplemented!();
                Ok(::frameworks::native::helper::NormalizationConfig)
            }

            fn lrn(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CLRN
            ) -> Result<(), ::co::error::Error> {
                unimplemented!();
                Ok(())
            }

            fn lrn_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CLRN
            ) -> Result<(), ::co::error::Error> {
                unimplemented!();
                Ok(())
            }

            fn lrn_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CLRN
            ) -> Result<(), ::co::error::Error> {
                unimplemented!();
                Ok(())
            }

            fn lrn_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &::co::tensor::SharedTensor<$t>,
                result: &::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CLRN
            ) -> Result<(), ::co::error::Error> {
                unimplemented!();
                Ok(())
            }
        }
    );
}

#[macro_export]
macro_rules! impl_ops_pooling_for {
    ($t:ident, $b:ty) => (
        impl ::plugin::Pooling<$t> for $b {
            fn new_pooling_config(
                &self,
                window: &[i32],
                padding: &[i32],
                stride: &[i32]
            ) -> Result<Self::CPOOL, ::co::error::Error> {
                unimplemented!();
                Ok(::frameworks::native::helper::PoolingConfig)
            }

            fn pooling_max(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CPOOL
            ) -> Result<(), ::co::error::Error> {
                unimplemented!();
                Ok(())
            }

            fn pooling_max_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CPOOL
            ) -> Result<(), ::co::error::Error> {
                unimplemented!();
                Ok(())
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
                unimplemented!();
                Ok(())
            }

            fn pooling_max_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &::co::tensor::SharedTensor<$t>,
                result: &::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CPOOL
            ) -> Result<(), ::co::error::Error> {
                unimplemented!();
                Ok(())
            }
        }
    );
}
