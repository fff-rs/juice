use std::fs;
use std::io;
use std::path;

use crate::net_capnp::network as capnp_network;

use crate::capnp_util::*;
use crate::co::frameworks::native::get_native_backend;
use crate::co::{IBackend, ITensorDesc, SharedTensor, TensorDesc};
use crate::coblas::plugin::Copy;
use crate::net::layer::Layer;
use crate::net::{layer_from_config, Context, Descriptor, Inout, LayerConfig};
use crate::util::LayerOps;

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
        backend.copy(&input, &mut context_inputs.borrow_mut()).unwrap();

        // Compute network output and take it out of the context as a return value.
        self.top.compute_output(backend, &mut context);
        context.take_data(self.top.descriptor().output(0))
    }

    pub fn save<P: AsRef<path::Path>>(&mut self, path: P) -> io::Result<()> {
        let path = path.as_ref();
        let ref mut out = fs::File::create(path)?;

        let mut message = ::capnp::message::Builder::new_default();
        {
            let mut net_message = message.init_root::<capnp_network::Builder>();
            self.write_capnp(&mut net_message);
        }
        ::capnp::serialize_packed::write_message(out, &message).unwrap();

        Ok(())
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
            backend.copy(&from_params.data, &mut to_params.data).unwrap();
            to_params.learning_rate = from_params.learning_rate;
        }

        net
    }
}

impl<'a, B: IBackend + LayerOps<f32>> CapnpWrite<'a> for Network<B> {
    type Builder = capnp_network::Builder<'a>;

    /// Write the Layer into a capnp message.
    fn write_capnp(&self, builder: &mut Self::Builder) {
        // Write top-level cofnig.
        {
            // let mut config_message = builder.reborrow().init_config();
            // self.config.write_capnp(&mut config_message);
        }

        // Write input shapes.
        {
            let input_shapes: Vec<TensorDesc> = self
                .top
                .descriptor()
                .inputs()
                .iter()
                .map(|input| input.unit_shape().clone())
                .collect();

            let inputs_count = self.top.descriptor().inputs().len();
            let mut inputs_message = builder.reborrow().init_inputs(input_shapes.len() as u32);
            for i in 0..input_shapes.len() {
                let mut input_message = inputs_message.reborrow().get(i as u32);
                let mut vals = input_message
                    .reborrow()
                    .init_shape(input_shapes[i].len() as u32);
                for j in 0..input_shapes[i].len() {
                    vals.set(j as u32, input_shapes[i][j] as u64);
                }
            }
        }
    }
}
