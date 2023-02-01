//! Applies a Dropout layer to the input data `x`
//!
//! The variables are:
//!
//! - `y`: output value
//! - `x`: input value
//! - `p`: dropout probability

use crate::capnp_util::*;
use crate::co::{IBackend, SharedTensor};
use crate::conn;
use crate::juice_capnp::dropout_config as capnp_config;
use crate::layer::*;
use crate::util::ArcLock;
use std::rc::Rc;

#[derive(Debug, Clone)]
/// [Dropout](./index.html) Layer
pub struct Dropout<T, B: conn::Dropout<T>> {
    probability: f32,
    seed: u64,
    dropout_config: Vec<Rc<B::CDROP>>,
}

impl<T, B: conn::Dropout<T>> Dropout<T, B> {
    /// Create a Dropout layer from a DropoutConfig.
    pub fn from_config(config: &DropoutConfig) -> Dropout<T, B> {
        Dropout {
            // TODO consider moving to vec
            probability: config.probability,
            // TODO consider moving to vec
            seed: config.seed,
            dropout_config: vec![],
        }
    }
}

//
// Dropout
//
impl<B: IBackend + conn::Dropout<f32>> ILayer<B> for Dropout<f32, B> {
    impl_ilayer_common!();

    fn reshape(
        &mut self,
        backend: ::std::rc::Rc<B>,
        input_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
        input_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
        weights_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
        weights_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
        output_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
        output_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
    ) {
        for i in 0..input_data.len() {
            let inp = input_data[0].read().unwrap();
            let input_desc = inp.desc();
            input_gradient[0].write().unwrap().resize(input_desc).unwrap();
            output_data[0].write().unwrap().resize(input_desc).unwrap();
            output_gradient[0].write().unwrap().resize(input_desc).unwrap();

            let config = backend.new_dropout_config(self.probability, self.seed).unwrap();
            self.dropout_config.push(Rc::new(config));
        }
    }
}

impl<B: IBackend + conn::Dropout<f32>> ComputeOutput<f32, B> for Dropout<f32, B> {
    fn compute_output(
        &self,
        backend: &B,
        _weights: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        output_data: &mut [&mut SharedTensor<f32>],
    ) {
        let config = &self.dropout_config[0];
        backend.dropout(input_data[0], output_data[0], &*config).unwrap();
    }
}

impl<B: IBackend + conn::Dropout<f32>> ComputeInputGradient<f32, B> for Dropout<f32, B> {
    fn compute_input_gradient(
        &self,
        backend: &B,
        weights_data: &[&SharedTensor<f32>],
        output_data: &[&SharedTensor<f32>],
        output_gradients: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        input_gradients: &mut [&mut SharedTensor<f32>],
    ) {
        let dropout_config = &self.dropout_config[0];
        backend
            .dropout_grad(
                output_data[0],
                output_gradients[0],
                input_data[0],
                input_gradients[0],
                dropout_config,
            )
            .unwrap()
    }
}

impl<B: IBackend + conn::Dropout<f32>> ComputeParametersGradient<f32, B> for Dropout<f32, B> {}

#[derive(Debug, Clone, PartialEq)]
/// Specifies configuration parameters for a Dropout Layer.
pub struct DropoutConfig {
    /// The probability to clamp a value to zero
    pub probability: f32,
    /// The initial seed for the (pseudo-)random generator
    pub seed: u64,
}

impl Into<LayerType> for DropoutConfig {
    fn into(self) -> LayerType {
        LayerType::Dropout(self)
    }
}

impl<'a> CapnpWrite<'a> for DropoutConfig {
    type Builder = capnp_config::Builder<'a>;

    /// Write the DropoutConfig into a capnp message.
    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.reborrow().set_probability(self.probability);
        builder.reborrow().set_seed(self.seed);
    }
}

impl<'a> CapnpRead<'a> for DropoutConfig {
    type Reader = capnp_config::Reader<'a>;

    fn read_capnp(reader: Self::Reader) -> Self {
        let probability: f32 = reader.get_probability();
        let seed: u64 = reader.get_seed();

        DropoutConfig {
            probability: probability,
            seed: seed,
        }
    }
}

impl ::std::default::Default for DropoutConfig {
    fn default() -> DropoutConfig {
        DropoutConfig {
            probability: 0.75,
            seed: 42,
        }
    }
}
