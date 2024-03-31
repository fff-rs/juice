use std::cell::RefCell;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;

use crate::co::{IBackend, ITensorDesc};
use crate::net::{Context, Descriptor, Layer, LayerError, LayerFromConfigError, LearnableParams};
use crate::util::LayerOps;
use crate::weight::FillerType;
use coaster::SharedTensor;
use conn::{DirectionMode, RnnAlgorithm, RnnInputMode, RnnNetworkMode};

#[derive(Clone, Debug, PartialEq)]
pub struct RnnConfig {
    /// Size of the hidden layer.
    pub hidden_size: usize,
    /// Number of hidden layers.
    pub num_layers: usize,
    /// Type of RNN.
    pub rnn_type: RnnNetworkMode,
    /// Dropout probability.
    pub dropout_probability: f32,
    /// Dropout seed
    pub dropout_seed: u64,
    /// Input mode.
    pub input_mode: RnnInputMode,
    /// RNN direction.
    pub direction_mode: DirectionMode,
}

pub struct Rnn<B: conn::Rnn<f32>> {
    descriptor: Descriptor,
    config: RnnConfig,

    // RNN config (batch size agnostic).
    rnn_context: RefCell<B::CRNN>,

    // RNN weights containing linear weights and biases for the RNN units.
    weights: Rc<RefCell<LearnableParams>>,
}

impl<B: conn::Rnn<f32>> Rnn<B> {
    pub fn new(backend: &B, mut descriptor: Descriptor, config: &RnnConfig) -> Result<Self, LayerFromConfigError> {
        if descriptor.inputs().len() != 1 {
            return Err(LayerFromConfigError::WrongInputs(format!(
                "Expected 1 input, got {}",
                descriptor.inputs().len()
            )));
        }

        let input_shape = descriptor.input(0).unit_shape();

        if input_shape.len() != 2 {
            return Err(LayerFromConfigError::WrongInputs(format!(
                "Input to RNN must be of [inputs_size, sequence_length] shape, got {:?}",
                input_shape
            )));
        }

        let input_size = input_shape[0];
        let sequence_length = input_shape[1];

        descriptor.add_output(vec![config.hidden_size, sequence_length]);

        let rnn_context = backend.new_rnn_config(
            Some(config.dropout_probability),
            Some(config.dropout_seed),
            sequence_length as i32,
            config.rnn_type,
            config.input_mode,
            config.direction_mode,
            // Standard is likely to be effective across most parameters. This should be
            // calculated internal to Juice if modified, allowing user input is likely to be
            // more confusing than helpful to the end user.
            // https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnRNNAlgo_t
            // lists the differences and how we can pick between algorithms automatically.
            RnnAlgorithm::Standard,
            input_size as i32,
            config.hidden_size as i32,
            config.num_layers as i32,
        )?;

        let weights_desc = backend
            .generate_rnn_weight_description(&rnn_context, input_size as i32)
            .unwrap();

        let mut weights_data = SharedTensor::new(&weights_desc);

        FillerType::fill_glorot(
            &mut weights_data,
            weights_desc.size(),
            config.num_layers * config.hidden_size,
        );

        let weights = descriptor.create_params("weights", weights_data, 1.0);

        Ok(Rnn {
            descriptor,
            config: config.clone(),
            rnn_context: RefCell::new(rnn_context),
            weights,
        })
    }
}

impl<B: IBackend + LayerOps<f32>> Layer<B> for Rnn<B> {
    fn compute_output(&self, backend: &B, context: &mut Context) -> Result<(), LayerError> {
        let input = context.get_data(self.descriptor.input(0));
        let output = context.acquire_data(self.descriptor.output(0));

        backend.rnn_forward(
            &input.borrow(),
            &mut output.borrow_mut(),
            &mut self.rnn_context.borrow_mut(),
            &self.weights.borrow().data,
        )?;
        Ok(())
    }

    fn compute_gradients(&self, backend: &B, context: &mut Context) -> Result<(), LayerError> {
        let input = context.get_data(self.descriptor.input(0));
        let output = context.get_data(self.descriptor.output(0));
        let output_gradient = context.get_data_gradient(self.descriptor.output(0));

        let input_gradient = context.acquire_data_gradient(self.descriptor.input(0));
        let weights_gradient = context.acquire_params_gradient(self.descriptor.param(0));

        backend.rnn_backward_data(
            &input.borrow(),
            &mut input_gradient.borrow_mut(),
            &output.borrow(),
            &output_gradient.borrow(),
            &mut self.rnn_context.borrow_mut(),
            &self.weights.borrow().data,
        )?;

        backend.rnn_backward_weights(
            &input.borrow(),
            &output.borrow(),
            &mut weights_gradient.borrow_mut(),
            &mut self.rnn_context.borrow_mut(),
        )?;

        Ok(())
    }

    fn descriptor(&self) -> &Descriptor {
        &self.descriptor
    }

    fn descriptor_mut(&mut self) -> &mut Descriptor {
        &mut self.descriptor
    }
}

impl<B: conn::Rnn<f32>> Debug for Rnn<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rnn").field("descriptor", &self.descriptor).finish()
    }
}
