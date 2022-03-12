use crate::co::frameworks::native::get_native_backend;
use crate::co::{IBackend, ITensorDesc, SharedTensor, TensorDesc};
use crate::net::{Context, LayerConfig, layer_from_config, Inout, Descriptor};
use crate::net::layer::Layer;
use crate::util::LayerOps;
use crate::coblas::plugin::Copy;

// A trainable network.
pub struct Network<B: IBackend + LayerOps<f32>> {
    config: LayerConfig,
    top: Box<dyn Layer<B>>,
}

impl<B: IBackend + LayerOps<f32> + 'static> Network<B> {
    /// Creates network from a config.
    pub fn from_config(config: LayerConfig, input_shapes: &[TensorDesc]) -> Network<B> {
        let inputs = input_shapes
            .iter()
            .enumerate()
            .map(|(i, shape)| Inout::new_with_path(shape.clone(), &format!("net_in_{}", i)))
            .collect();
        let descriptor = Descriptor::top("net", inputs);
        let top = layer_from_config(descriptor, &config);

        Network {
            config: config,
            top: top,
        }
    }

    /// Top level layer (typically a container layer).
    pub fn top(&self) -> &dyn Layer<B> {
        self.top.as_ref()
    }

    /// Mutable top level layer (typically a container layer).
    pub fn top_mut(&mut self) -> &mut dyn Layer<B> {
        self.top.as_mut()
    }

    /// Do a forward pass on the provided inputs and return the network output.
    /// This function assumes the network has exactly one input and exactly one output
    /// (will panic otherwise).
    /// Input shape must be either [<top input shape>] or [N, <top input shape>]
    /// (latter case for batched processing).
    /// Returns a tensor of shape which is either [<top output shape>] or [N, <top output shape>],
    /// depending on the input shape.
    pub fn transform(&self, backend: &B, input: &SharedTensor<f32>) -> SharedTensor<f32> {
        assert_eq!(self.top.descriptor().inputs().len(), 1);
        assert_eq!(self.top.descriptor().outputs().len(), 1);

        // Figure out batch size.
        let net_input_size = self.top.descriptor().input(0).unit_shape().size();
        let batch_size = match input.desc().size() {
            net_input_size => 1,
            _ => {
                assert!(input.desc().len() > 1);
                let input_unit_size = input.desc().iter().skip(1).fold(1, |acc, i| acc * i);
                assert_eq!(input_unit_size, net_input_size);
                input.desc()[0]
            }
        };

        let mut context = Context::new(batch_size);

        // Copy input data into the context.
        let context_inputs = context.acquire_data(self.top.descriptor().input(0));
        assert_eq!(context_inputs.borrow().desc().size(), input.desc().size());
        backend.copy(&input, &mut context_inputs.borrow_mut());

        // Compute network output and take it out of the context as a return value.
        self.top.compute_output(backend, &mut context);
        context.take_data(self.top.descriptor().output(0))
    }
}

impl<B: IBackend + LayerOps<f32> + 'static> Clone for Network<B> {
    fn clone(&self) -> Network<B> {
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
            backend.copy(&from_params.data, &mut to_params.data);
            to_params.learning_rate = from_params.learning_rate;
        }

        net
    }
}
