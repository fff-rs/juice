use crate::co::frameworks::native::get_native_backend;
use crate::co::{IBackend, ITensorDesc, SharedTensor, TensorDesc};
use crate::coblas::plugin::Copy;
use crate::net::layer::Layer;
use crate::net::{layer_from_config, Context, Descriptor, Inout, LayerConfig};
use crate::util::LayerOps;

// A trainable network. Essentially a convenience wrapper around the top-level layer
// which is typically a container layer.
pub struct Network {
    // Configuration of the top layer.
    config: LayerConfig,
    // Top layer.
    top: Box<dyn Layer>,
}

impl Network {
    /// Creates network from a config with the given input shapes.
    pub fn from_config(config: LayerConfig, input_shapes: &[TensorDesc]) -> Network {
        let inputs = input_shapes
            .iter()
            .enumerate()
            .map(|(i, shape)| {
                let mut input = Inout::new(shape.clone());
                input.set_path(&format!("net_in_{}", i));
                input
            })
            .collect();
        let descriptor = Descriptor::top("net", inputs);
        let top = layer_from_config(descriptor, &config);

        Network {
            config,
            top,
        }
    }

    pub fn top(&self) -> &dyn Layer {
        self.top.as_ref()
    }

    pub fn top_mut(&mut self) -> &mut dyn Layer {
        self.top.as_mut()
    }

    /// Does a forward pass on the provided inputs and returns the network output.
    /// This is the main function to use the network after training.
    /// Assumes the network has exactly one input and exactly one output (will panic otherwise).
    /// Input shape must be either [<top input shape>] or [N, <top input shape>]
    /// (latter case for batch processing).
    /// Returns a tensor of shape which is either [<top output shape>] or [N, <top output shape>],
    /// depending on the input shape.
    pub fn transform(&self, backend: &B, input: &SharedTensor<f32>) -> SharedTensor<f32> {
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
        let context_inputs = context.acquire_data(self.top.descriptor().input(0));
        assert_eq!(context_inputs.borrow().desc().size(), input.desc().size());
        backend
            .copy(&input, &mut context_inputs.borrow_mut())
            .unwrap();

        // Compute network output and take it out of the context as a return value.
        self.top.compute_output(backend, &mut context);
        context.take_data(self.top.descriptor().output(0))
    }
}

impl Clone for Network {
    fn clone(&self) -> Network {
        let input_shapes: Vec<TensorDesc> = self
            .top
            .descriptor()
            .inputs()
            .iter()
            .map(|input| input.unit_shape().clone())
            .collect();
        let net = Network::from_config(self.config.clone(), &input_shapes);

        // Copy weights data.
        let backend = get_native_backend();
        assert_eq!(
            self.top.descriptor().params().len(),
            net.top.descriptor().params().len()
        );
        for i in 0..self.top.descriptor().params().len() {
            let from_params = self.top.descriptor().params()[i].borrow();
            let mut to_params = net.top.descriptor().params()[i].borrow_mut();
            backend
                .copy(&from_params.data, &mut to_params.data)
                .unwrap();
            to_params.learning_rate = from_params.learning_rate;
        }

        net
    }
}