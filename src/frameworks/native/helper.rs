//! Provides useful macros for easier NN implementation for native.

use co;
use co::plugin::numeric_helpers::Float;
use co::memory::MemoryType;
use co::plugin::Error as PluginError;

#[derive(Debug, Copy, Clone)]
#[allow(missing_docs)]
pub struct ConvolutionConfig;
#[derive(Debug, Copy, Clone)]
#[allow(missing_docs)]
pub struct NormalizationConfig;
#[derive(Debug, Copy, Clone)]
#[allow(missing_docs)]
pub struct PoolingConfig;

macro_rules! read {
    ($x:ident, $t:ident, $slf:ident) => (
        try!($x.read($slf.device())).as_native()
            .expect("Broken invariant: not a CUDA memory")
            .as_slice::<$t>()
    )
}

macro_rules! read_write {
    ($x:ident, $t: ident, $slf:ident) => (
        try!($x.read_write($slf.device())).as_mut_native()
            .expect("Broken invariant: not a CUDA memory")
            .as_mut_slice::<$t>()
    )
}

macro_rules! write_only {
    ($x:ident, $t: ident, $slf:ident) => (
        try!($x.write_only($slf.device())).as_mut_native()
            .expect("Broken invariant: not a CUDA memory")
            .as_mut_slice::<$t>()
    )
}

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
pub fn sigmoid<T: Float>(x: T) -> T {
    (T::one()) / (T::one() + (-x).exp())
}

#[inline]
/// Computes the Sigmoid Gradient on the CPU
pub fn sigmoid_grad<T: Float>(x: T, dx: T) -> T {
    x * (T::one() - x) * dx
}

#[inline]
/// Computes the ReLU Function on the CPU
pub fn relu<T: Float>(x: T) -> T {
    let x : T = x.clone();
    x.max(T::zero())
}

#[inline]
/// Computes the ReLU Gradient on the CPU
pub fn relu_grad<T: Float>(x: T, dx: T) -> T {
    if x > T::zero() {
        return dx
    }
    T::zero()
}

#[inline]
/// Computes the Tanh Function on the CPU
pub fn tanh<T: Float>(x: T) -> T {
    x.tanh()
}

#[inline]
// d/dx tanh x = sech2 x = 1 + tanh2 x
/// Computes the Tanh Gradient on the CPU
pub fn tanh_grad<T: Float>(x: T, dx: T) -> T {
    (T::one() - x.powi(2)) * dx
}

macro_rules! impl_oconf_for_cc(($($t: ident), +) => (
    $(
        impl<'a> NNOperationConfig<$t> for ::frameworks::native::helper::ConvolutionConfig { }
        impl<'a> ConvolutionConfig<$t> for ::frameworks::native::helper::ConvolutionConfig { }
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
        impl Sigmoid<$t> for $b {
            fn sigmoid(&self, x: &SharedTensor<$t>, result: &mut SharedTensor<$t>)
                       -> Result<(), Error> {
                map1(read!(x, $t, self),
                     write_only!(result, $t, self),
                     ::frameworks::native::helper::sigmoid)
            }

            fn sigmoid_grad(
                &self,
                x: &SharedTensor<$t>,
                x_diff: &SharedTensor<$t>,
                result: &SharedTensor<$t>,
                result_diff: &mut SharedTensor<$t>)
                -> Result<(), Error> {
                map2(read!(x, $t, self),
                     read!(x_diff, $t, self),
                     write_only!(result_diff, $t, self),
                     ::frameworks::native::helper::sigmoid_grad)
            }
        }

        impl SigmoidPointwise<$t> for $b {
            fn sigmoid_pointwise(&self, x: &mut SharedTensor<$t>)
                       -> Result<(), ::co::error::Error> {
                map1_inplace(read_write!(x, $t, self),
                     ::frameworks::native::helper::sigmoid)
            }

            fn sigmoid_pointwise_grad(
                &self,
                x: &SharedTensor<$t>,
                x_diff: &mut SharedTensor<$t>)
                -> Result<(),  ::co::error::Error> {
                    return
                map2_inplace(read!(x, $t, self),
                     read_write!(x_diff, $t, self),
                     ::frameworks::native::helper::sigmoid_grad)
            }
        }
    );
}

#[macro_export]
macro_rules! impl_ops_relu_for {
    ($t:ident, $b:ty) => (
        impl Relu<$t> for $b {
            fn relu(&self, x: &SharedTensor<$t>, result: &mut SharedTensor<$t>)
                    -> Result<(), ::co::error::Error> {
                map1(read!(x, $t, self),
                     write_only!(result, $t, self),
                     ::frameworks::native::helper::relu)
            }

            fn relu_grad(
                &self,
                x: &SharedTensor<$t>,
                x_diff: &SharedTensor<$t>,
                result: &SharedTensor<$t>,
                result_diff: &mut SharedTensor<$t>)
                -> Result<(), Error> {
                map2(read!(x, $t, self),
                     read!(x_diff, $t, self),
                     write_only!(result_diff, $t, self),
                     ::frameworks::native::helper::relu_grad)
            }
        }
        impl ReluPointwise<$t> for $b {

            fn relu_pointwise(&self, x: &mut SharedTensor<$t>)
                    -> Result<(), ::co::error::Error> {
                map1_inplace(read_write!(x, $t, self),
                     ::frameworks::native::helper::relu)
            }

            fn relu_pointwise_grad(
                &self,
                x: &SharedTensor<$t>,
                x_diff: &mut SharedTensor<$t>)
                -> Result<(), ::co::error::Error> {
                map2_inplace(read!(x, $t, self),
                     read_write!(x_diff, $t, self),
                     ::frameworks::native::helper::relu_grad)
            }
        }
    );
}

#[macro_export]
macro_rules! impl_ops_tanh_for {
    ($t:ident, $b:ty) => (
        impl ::plugin::Tanh<$t> for $b {
            fn tanh(&self, x: &SharedTensor<$t>, result: &mut SharedTensor<$t>)
                    -> Result<(), ::co::error::Error> {
                map1(read!(x, $t, self),
                     write_only!(result, $t, self),
                     ::frameworks::native::helper::tanh)
            }

            fn tanh_grad(
                &self,
                x: &SharedTensor<$t>,
                x_diff: &SharedTensor<$t>,
                result: &SharedTensor<$t>,
                result_diff: &mut SharedTensor<$t>)
                -> Result<(), Error> {
                map2(read!(x, $t, self),
                     read!(x_diff, $t, self),
                     write_only!(result_diff, $t, self),
                     ::frameworks::native::helper::tanh_grad)
            }
        }
        impl ::plugin::TanhPointwise<$t> for $b {
            fn tanh_pointwise(&self, x: &mut SharedTensor<$t>)
                    -> Result<(), ::co::error::Error> {
                map1_inplace(read_write!(x, $t, self),
                     ::frameworks::native::helper::tanh)
            }

            fn tanh_pointwise_grad(
                &self,
                x: &SharedTensor<$t>,
                x_diff: &mut SharedTensor<$t>)
                -> Result<(), Error> {
                map2_inplace(read!(x, $t, self),
                     read_write!(x_diff, $t, self),
                     ::frameworks::native::helper::tanh_grad)
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
                src: &SharedTensor<$t>,
                dest: &SharedTensor<$t>,
                filter: &mut SharedTensor<$t>,
                stride: &[i32],
                zero_padding: &[i32]
            ) -> Result<Self::CC, ::co::error::Error> {
                unimplemented!();
                Ok(helper::ConvolutionConfig)
            }

            fn convolution(
                &self,
                x: &SharedTensor<$t>,
                result: &mut SharedTensor<$t>,
                config: &Self::CC
            ) -> Result<(), ::co::error::Error> {
                unimplemented!();
                Ok(())
            }

            fn convolution_grad(
                &self,
                x: &SharedTensor<$t>,
                x_diff: &SharedTensor<$t>,
                result: &SharedTensor<$t>,
                result_diff: &mut SharedTensor<$t>,
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
            fn softmax(&self, x: &SharedTensor<$t>, result: &mut SharedTensor<$t>)
                       -> Result<(), Error> {
                let xs = read!(x, $t, self);
                let rs = write_only!(result, $t, self);

                try!(map1(xs, rs, |v| v.exp()));

                let mut sum: $t = 0.0; // iter_arith is not stable yet
                for r in &*rs {
                    sum += *r;
                }
                for r in rs {
                    *r /= sum;
                }
                Ok(())
            }

            // TODO: check
            fn softmax_grad(
                &self,
                x: &SharedTensor<$t>,
                x_diff: &SharedTensor<$t>,
                result_diff: &mut SharedTensor<$t>) -> Result<(), Error> {

                let xs = read!(x, $t, self);
                let dxs = read!(x_diff, $t, self);
                let drs = write_only!(result_diff, $t, self);

                let mut dot: $t = 0 as $t;
                for (t, dt) in xs.iter().zip(dxs.iter()) {
                    dot += t * dt;
                }

                map2(xs, dxs, drs, |t, dt| t * (dt - dot))
            }
        }
    );
}

#[macro_export]
macro_rules! impl_ops_log_softmax_for {
    ($t:ident, $b:ty) => (
        impl ::plugin::LogSoftmax<$t> for $b {
            fn log_softmax(&self, x: &SharedTensor<$t>, result: &mut SharedTensor<$t>)
                           -> Result<(), ::co::error::Error> {
                let xs = read!(x, $t, self);
                let rs = write_only!(result, $t, self);

                let max_x = xs.iter().fold(::std::$t::NEG_INFINITY,
                                           |acc, &t| acc.max(t));

                let mut logsum : $t = 0 as $t;
                for t in xs {
                    logsum += (-(max_x - t)).exp();
                }
                logsum = max_x + logsum.ln();

                map1(xs, rs, |t| t - logsum)
            }

            fn log_softmax_grad(&self, x: &SharedTensor<$t>, x_diff: &SharedTensor<$t>,
                                result_diff: &mut SharedTensor<$t>)
                                -> Result<(), ::co::error::Error> {
                let xs = read!(x, $t, self);
                let dxs = read!(x_diff, $t, self);
                let drs = write_only!(result_diff, $t, self);

                let mut sum = 0 as $t;
                for &grad_val in dxs.iter() {
                    sum += grad_val;
                }
                map2(xs, dxs, drs, |t, dt| dt - t.exp() * sum)
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
                x: &mut SharedTensor<$t>,
                result: &mut SharedTensor<$t>,
                config: &Self::CLRN
            ) -> Result<(), ::co::error::Error> {
                unimplemented!();
                Ok(())
            }

            fn lrn_plain(
                &self,
                x: &SharedTensor<$t>,
                result: &mut SharedTensor<$t>,
                config: &Self::CLRN
            ) -> Result<(), ::co::error::Error> {
                unimplemented!();
                Ok(())
            }

            fn lrn_grad(
                &self,
                x: &mut SharedTensor<$t>,
                x_diff: &mut SharedTensor<$t>,
                result: &mut SharedTensor<$t>,
                result_diff: &mut SharedTensor<$t>,
                config: &Self::CLRN
            ) -> Result<(), ::co::error::Error> {
                unimplemented!();
                Ok(())
            }

            fn lrn_grad_plain(
                &self,
                x: &SharedTensor<$t>,
                x_diff: &SharedTensor<$t>,
                result: &SharedTensor<$t>,
                result_diff: &mut SharedTensor<$t>,
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
                x: &mut SharedTensor<$t>,
                result: &mut SharedTensor<$t>,
                config: &Self::CPOOL
            ) -> Result<(), ::co::error::Error> {
                unimplemented!();
                Ok(())
            }

            fn pooling_max_plain(
                &self,
                x: &SharedTensor<$t>,
                result: &mut SharedTensor<$t>,
                config: &Self::CPOOL
            ) -> Result<(), ::co::error::Error> {
                unimplemented!();
                Ok(())
            }
            #[allow(unused_variables)]
            fn pooling_max_grad(
                &self,
                x: &mut SharedTensor<$t>,
                x_diff: &mut SharedTensor<$t>,
                result: &mut SharedTensor<$t>,
                result_diff: &mut SharedTensor<$t>,
                config: &Self::CPOOL
            ) -> Result<(), ::co::error::Error> {
                unimplemented!();
                Ok(())
            }

            fn pooling_max_grad_plain(
                &self,
                x: &SharedTensor<$t>,
                x_diff: &SharedTensor<$t>,
                result: &SharedTensor<$t>,
                result_diff: &mut SharedTensor<$t>,
                config: &Self::CPOOL
            ) -> Result<(), ::co::error::Error> {
                unimplemented!();
                Ok(())
            }
        }
    );
}
