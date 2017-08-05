//! Applies a Dropout layer to the input data `x`
//!
//! The variables are:
//!
//! - `y`: output value
//! - `x`: input value
//! - `p`: dropout probability

use super::FilterLayer;
use capnp_util::*;
use co::{IBackend, SharedTensor};
use conn;
use layer::*;
use juice_capnp::dropout_config as capnp_config;
use std::rc::Rc;
use util::ArcLock;

#[derive(Debug, Clone)]
/// [Dropout](./index.html) Layer
pub struct Dropout<T, B: conn::Dropout<T>> {
    probability: f32,
    dropout_config: Option<Rc<B::CDROP>>,
}


impl<T, B: conn::Dropout<T>> Dropout<T, B> {
    /// Create a Dropout layer from a DropoutConfig.
    pub fn from_config(config: &DropoutConfig) -> Dropout<T, B> {
        Dropout {
            probability: config.probability,
            dropout_config: None,
        }
    }
}

//
// Dropout
//
impl<B: IBackend + conn::Dropout<f32>> ILayer<B> for Dropout<f32, B> {
    impl_ilayer_common!();

    fn reshape(&mut self,
               backend: ::std::rc::Rc<B>,
               input_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               input_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               output_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               output_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>) {
        if let Some(inp) = input_data.get(0) {
            let read_inp = inp.read().unwrap();
            let input_desc = read_inp.desc();
            input_gradient[0].write().unwrap().resize(input_desc).unwrap();
            output_data[0].write().unwrap().resize(input_desc).unwrap();
            output_gradient[0].write().unwrap().resize(input_desc).unwrap();
        }
    }
}

impl<B: IBackend + conn::Dropout<f32>> ComputeOutput<f32, B> for Dropout<f32,B> {
    fn compute_output(&self,
                      backend: &B,
                      _weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {

        let dropout_config = self.dropout_config.as_ref().unwrap();
	backend.dropout(input_data[0], output_data[0], dropout_config).unwrap();
    }
}

impl<B: IBackend + conn::Dropout<f32>> ComputeInputGradient<f32, B> for Dropout<f32,B> {
    fn compute_input_gradient(&self,
                              backend: &B,
                              weights_data: &[&SharedTensor<f32>],
                              output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {

        let dropout_config = self.dropout_config.as_ref().unwrap();
        backend.dropout_grad(output_data[0],
                       output_gradients[0],
                       input_data[0],
                       input_gradients[0],
                       dropout_config)
            .unwrap()
    }
}

impl<B: IBackend + conn::Dropout<f32>> ComputeParametersGradient<f32, B> for Dropout<f32,B> {}

#[derive(Debug, Copy, Clone)]
/// Specifies configuration parameters for a Dropout Layer.
pub struct DropoutConfig {
    /// The probability to clamp a value to zero
    pub probability: f32,
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
        builder.borrow().set_dropout(self.probability);

    }
}

impl<'a> CapnpRead<'a> for DropoutConfig {
    type Reader = capnp_config::Reader<'a>;

    fn read_capnp(reader: Self::Reader) -> Self {
        let probability : f32 = reader.get_dropout();

        DropoutConfig {
            probability: probability,
        }
    }
}
