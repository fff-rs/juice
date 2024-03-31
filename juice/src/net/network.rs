use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;

use serde::{Deserialize, Serialize};

use crate::co::frameworks::native::get_native_backend;
use crate::co::{IBackend, ITensorDesc, SharedTensor, TensorDesc};
use crate::coblas::plugin::Copy;
use crate::net::layer::Layer;
use crate::net::{layer_from_config, Context, Descriptor, Inout, LayerConfig};
use crate::util::{native_backend, LayerOps};

use super::{LayerError, LayerFromConfigError};

/// A trainable network. Essentially a convenience wrapper around the top-level layer
/// which is typically a container layer.
pub struct Network<B: IBackend + LayerOps<f32>> {
    // Configuration of the top layer.
    config: LayerConfig,
    // Top layer.
    top: Box<dyn Layer<B>>,
}

/// Representation of all weights in the model that can be serialized and deserialized.
#[derive(Serialize, Deserialize)]
pub struct WeightsData {
    /// Maps weights path in the network (`LearnableParams.path`) to weights tensor data.
    pub weights_data: HashMap<String, Vec<f32>>,
}

/// Errors that can happen during network creation.
#[derive(Debug, thiserror::Error)]
pub enum NetworkFromConfigError {
    #[error("Layer creation failed: {0}")]
    Layer(#[from] LayerFromConfigError),
    #[error("Provided weights don't have data for param {0}")]
    MissingWeights(String),
    #[error("Provided weights contain params not in the network: {0:?}. Are they from a different network?")]
    UnusedWeights(Vec<String>),
    #[error("Provided weights have wrong size for {path}; expected {expected}, got {actual}")]
    WeightsSizeMismatch {
        path: String,
        expected: usize,
        actual: usize,
    },
    #[error("Tensor operation error: {0}")]
    Tensor(#[from] co::tensor::Error),
}

impl<B: IBackend + LayerOps<f32> + 'static> Network<B> {
    /// Creates network from a config with the given input shapes.
    pub fn from_config(
        backend: &B,
        into_config: impl Into<LayerConfig>,
        input_shapes: &[TensorDesc],
    ) -> Result<Network<B>, NetworkFromConfigError> {
        let inputs = input_shapes
            .iter()
            .enumerate()
            .map(|(i, shape)| {
                let input = Inout::new(shape.clone());
                input.junction.path.replace(format!("net_in_{}", i));
                input
            })
            .collect();
        let descriptor = Descriptor::top("net", inputs);
        let config = into_config.into();
        let top = layer_from_config(backend, descriptor, &config)?;

        Ok(Network { config, top })
    }

    /// Creates network from a config and weights. Weights must comes from a network
    /// created with the same config and input shapes (typically acquired via
    /// copy_weights_data() and maybe serialized to a file).
    pub fn from_config_and_weights(
        backend: &B,
        into_config: impl Into<LayerConfig>,
        input_shapes: &[TensorDesc],
        weights: &WeightsData,
    ) -> Result<Network<B>, NetworkFromConfigError> {
        let net = Network::from_config(backend, into_config, input_shapes)?;

        // Set weights in the network from the provided data.
        let native_backend = native_backend();
        let mut unused_weights = HashSet::<String>::from_iter(weights.weights_data.keys().cloned());
        for i in 0..net.top.descriptor().params().len() {
            let mut to_params = net.top.descriptor().params()[i].borrow_mut();
            match weights.weights_data.get(&to_params.path) {
                Some(d) => {
                    let params_data = to_params.data.write_only(native_backend.device())?;
                    params_data.as_mut_slice::<f32>().copy_from_slice(d);

                    unused_weights.remove(&to_params.path);
                }
                None => return Err(NetworkFromConfigError::MissingWeights(to_params.path.clone())),
            }
        }

        // Check if all weights were used.
        if !unused_weights.is_empty() {
            return Err(NetworkFromConfigError::UnusedWeights(Vec::from_iter(unused_weights)));
        }

        Ok(net)
    }

    pub fn top(&self) -> &dyn Layer<B> {
        self.top.as_ref()
    }

    pub fn top_mut(&mut self) -> &mut dyn Layer<B> {
        self.top.as_mut()
    }

    /// Does a forward pass on the provided inputs and returns the network output.
    /// This is the main function to use the network after training.
    /// Assumes the network has exactly one input and exactly one output (will panic otherwise).
    /// Input shape must be either [<top input shape>] or [N, <top input shape>]
    /// (latter case for batch processing).
    /// Returns a tensor of shape which is either [<top output shape>] or [N, <top output shape>],
    /// depending on the input shape.
    pub fn transform(&self, backend: &B, input: &SharedTensor<f32>) -> Result<SharedTensor<f32>, LayerError> {
        assert_eq!(self.top.descriptor().inputs().len(), 1);
        assert_eq!(self.top.descriptor().outputs().len(), 1);

        // Figure out the batch size.
        let net_input_size = self.top.descriptor().input(0).unit_shape().size();
        let batch_size = if input.desc().size() == net_input_size {
            1
        } else {
            assert!(input.desc().len() > 1);
            let input_unit_size = input.desc().iter().skip(1).fold(1, |acc, i| acc * i);
            assert_eq!(input_unit_size, net_input_size);
            input.desc()[0]
        };

        let mut context = Context::new(batch_size);

        // Copy input data into the context.
        {
            let context_inputs = context.acquire_data(self.top.descriptor().input(0));
            assert_eq!(context_inputs.borrow().desc().size(), input.desc().size());
            backend.copy(&input, &mut context_inputs.borrow_mut()).unwrap();
        }

        // Compute network output and take it out of the context as a return value.
        self.top.compute_output(backend, &mut context)?;
        Ok(context.take_data(self.top.descriptor().output(0)))
    }

    /// Copies weights data into a serializable format.
    pub fn copy_weights_data(&self) -> WeightsData {
        let native_backend = native_backend();
        WeightsData {
            weights_data: self
                .top
                .descriptor()
                .params()
                .iter()
                .map(|pr| {
                    let p = pr.borrow();
                    (
                        p.path.clone(),
                        p.data.read(native_backend.device()).unwrap().as_slice().to_vec(),
                    )
                })
                .collect(),
        }
    }

    pub fn clone(&self, backend: &B) -> Network<B> {
        let input_shapes: Vec<TensorDesc> = self
            .top
            .descriptor()
            .inputs()
            .iter()
            .map(|input| input.unit_shape().clone())
            .collect();
        let net = Network::from_config(backend, self.config.clone(), &input_shapes).unwrap();

        // Copy weights data.
        let backend = get_native_backend();
        assert_eq!(
            self.top.descriptor().params().len(),
            net.top.descriptor().params().len()
        );
        for i in 0..self.top.descriptor().params().len() {
            let from_params = self.top.descriptor().params()[i].borrow();
            let mut to_params = net.top.descriptor().params()[i].borrow_mut();
            backend.copy(&from_params.data, &mut to_params.data).unwrap();
            to_params.learning_rate = from_params.learning_rate;
        }

        net
    }
}
