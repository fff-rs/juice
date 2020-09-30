//! Provides common utility functions

use std::rc::Rc;

use crate::co::frameworks::native::flatbox::FlatBox;
use crate::co::prelude::*;
use crate::coblas::plugin::*;
use crate::conn;
use crate::layer::{LayerType,LayerConfig,ILayer};
use num::traits::{cast, NumCast};
use std::sync::{Arc, RwLock};

/// Shared Lock used for our tensors
pub type ArcLock<T> = Arc<RwLock<T>>;

/// Create a simple native backend.
///
/// This is handy when you need to sync data to host memory to read/write it.
pub fn native_backend() -> Backend<Native> {
    let framework = Native::new();
    let hardwares = &framework.hardwares().to_vec();
    let backend_config = BackendConfig::new(framework, hardwares);
    Backend::new(backend_config).unwrap()
}

/// Write into a native Coaster Memory.
pub fn write_to_memory<T: NumCast + ::std::marker::Copy>(mem: &mut FlatBox, data: &[T]) {
    write_to_memory_offset(mem, data, 0);
}

/// Write into a native Coaster Memory with a offset.
pub fn write_to_memory_offset<T: NumCast + ::std::marker::Copy>(mem: &mut FlatBox, data: &[T], offset: usize) {
    let mem_buffer = mem.as_mut_slice::<f32>();
    for (index, datum) in data.iter().enumerate() {
        // mem_buffer[index + offset] = *datum;
        mem_buffer[index + offset] = cast(*datum).unwrap();
    }
}

/// Write the `i`th sample of a batch into a SharedTensor.
///
/// The size of a single sample is inferred through
/// the first dimension of the SharedTensor, which
/// is assumed to be the batchsize.
///
/// Allocates memory on a Native Backend if neccessary.
pub fn write_batch_sample<T: NumCast + ::std::marker::Copy>(
    tensor: &mut SharedTensor<f32>,
    data: &[T], i: usize) {
    let native_backend = native_backend();
    let tensor_desc = tensor.desc();
    let batch_size = tensor_desc[0];
    let batch_sample_size = tensor_desc.size();
    let sample_size = batch_sample_size / batch_size;

    write_to_memory_offset(
        tensor.write_only(native_backend.device()).unwrap(),
        &data,
        i * sample_size,
    );
}

/// Create a Coaster SharedTensor for a scalar value.
pub fn native_scalar<T: NumCast + ::std::marker::Copy>(scalar: T) -> SharedTensor<T> {
    let native = native_backend();
    let mut shared_scalar = SharedTensor::<T>::new(&[1]);
    write_to_memory(shared_scalar.write_only(native.device()).unwrap(), &[scalar]);
    shared_scalar
}

/// Casts a Vec<usize> to as Vec<i32>
pub fn cast_vec_usize_to_i32(input: Vec<usize>) -> Vec<i32> {
    let mut out = Vec::new();
    for i in input.iter() {
        out.push(*i as i32);
    }
    out
}

/// Extends IBlas with Axpby
pub trait Axpby<F>: Axpy<F> + Scal<F> {
    /// Performs the operation y := a*x + b*y .
    ///
    /// Consists of a scal(b, y) followed by a axpby(a,x,y).
    fn axpby(
        &self,
        a: &SharedTensor<F>,
        x: &SharedTensor<F>,
        b: &SharedTensor<F>,
        y: &mut SharedTensor<F>,
    ) -> Result<(), crate::co::error::Error> {
        self.scal(b, y)?;
        self.axpy(a, x, y)?;
        Ok(())
    }
}

impl<T: Axpy<f32> + Scal<f32>> Axpby<f32> for T {}

/// Encapsulates all traits required by Solvers.
pub trait SolverOps<F>: Axpby<F> + Dot<F> + Copy<F> {}

impl<T: Axpby<f32> + Dot<f32> + Copy<f32>> SolverOps<f32> for T {}


use crate::layers::*;


/// Encapsulates all traits used in Layers per `Framework` and data type `F`.
pub trait LayerOps<Framework: Clone + IFramework, F> : coblas::plugin::Copy<f32>
{
    /// Helper for [from_config] to match a [LayerType][2] to its [implementation][3].
    /// [1]: #method.from_config
    /// [2]: ./enum.LayerType.html
    /// [3]: ../layers/index.html
    fn layer_from_config<B: IBackend<F=Framework>>(backend: Rc<B>, config: &LayerConfig) -> Box<dyn ILayer<B>>;
}

#[cfg(feature = "native")]
impl<F,T> LayerOps<co::frameworks::Native, F> for T where
    T: conn::Convolution<f32>
        + conn::Rnn<f32>
        + conn::Pooling<f32>
        + conn::Relu<f32>
        + conn::ReluPointwise<f32>
        + conn::Sigmoid<f32>
        + conn::SigmoidPointwise<f32>
        + conn::Tanh<f32>
        + conn::TanhPointwise<f32>
        + conn::Softmax<f32>
        + conn::LogSoftmax<f32>
        + conn::Dropout<f32>
        + Gemm<f32>
        + Axpby<f32>
        + Copy<f32>
{
    fn layer_from_config<B: IBackend<F=co::frameworks::Native>>(backend: Rc<B>, config: &LayerConfig) -> Box<dyn ILayer<B>> {
        match config.layer_type {
            LayerType::Linear(layer_config) => Box::new(Linear::from_config(&layer_config)),
            LayerType::LogSoftmax => Box::new(LogSoftmax::default()),
            LayerType::Pooling(layer_config) => Box::new(Pooling::from_config(&layer_config)),
            LayerType::Sequential(layer_config) => Box::new(Sequential::from_config(backend, &layer_config)),
            LayerType::Softmax => Box::new(Softmax::default()),
            LayerType::ReLU => Box::new(ReLU),
            LayerType::TanH => Box::new(TanH),
            LayerType::Sigmoid => Box::new(Sigmoid),
            LayerType::NegativeLogLikelihood(layer_config) => {
                Box::new(NegativeLogLikelihood::from_config(&layer_config))
            }
            LayerType::MeanSquaredError => Box::new(MeanSquaredError),
            LayerType::Reshape(layer_config) => Box::new(Reshape::from_config(&layer_config)),
            LayerType::Dropout(layer_config) => Box::new(Dropout::from_config(&layer_config)),
            unsupported => panic!("Native does not support the requested layer type {:?}", unsupported),
        }
    }
}

#[cfg(feature = "cuda")]
impl<F,T> LayerOps<co::frameworks::Cuda, F> for T where
    T: conn::Convolution<f32>
        + conn::Rnn<f32>
        + conn::Pooling<f32>
        + conn::Relu<f32>
        + conn::ReluPointwise<f32>
        + conn::Sigmoid<f32>
        + conn::SigmoidPointwise<f32>
        + conn::Tanh<f32>
        + conn::TanhPointwise<f32>
        + conn::Softmax<f32>
        + conn::LogSoftmax<f32>
        + conn::Dropout<f32>
        + Gemm<f32>
        + Axpby<f32>
        + Copy<f32>
{
    fn layer_from_config<B: IBackend<F=co::frameworks::Cuda>>(backend: Rc<B>, config: &LayerConfig) -> Box<dyn ILayer<B>> {
    // fn layer_from_config(backend: Rc<Backend<co::frameworks::Cuda>>, config: &LayerConfig) -> Box<dyn ILayer<Backend<co::frameworks::Cuda>>> {
        match config.layer_type {
            LayerType::Convolution(layer_config) => Box::new(Convolution::from_config(layer_config)),
            LayerType::Rnn(layer_config) => Box::new(Rnn::from_config(&layer_config)),
            LayerType::Linear(layer_config) => Box::new(Linear::from_config(&layer_config)),
            LayerType::LogSoftmax => Box::new(LogSoftmax::default()),
            LayerType::Pooling(layer_config) => Box::new(Pooling::from_config(&layer_config)),
            LayerType::Sequential(layer_config) => Box::new(Sequential::from_config(backend, &layer_config)),
            LayerType::Softmax => Box::new(Softmax::default()),
            LayerType::ReLU => Box::new(ReLU),
            LayerType::TanH => Box::new(TanH),
            LayerType::Sigmoid => Box::new(Sigmoid),
            LayerType::NegativeLogLikelihood(layer_config) => {
                Box::new(NegativeLogLikelihood::from_config(&layer_config))
            }
            LayerType::MeanSquaredError => Box::new(MeanSquaredError),
            LayerType::Reshape(layer_config) => Box::new(Reshape::from_config(&layer_config)),
            LayerType::Dropout(layer_config) => Box::new(Dropout::from_config(&layer_config)),
        }
    }

}
