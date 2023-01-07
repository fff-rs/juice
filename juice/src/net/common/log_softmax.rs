use crate::co::IBackend;
use crate::conn;
use crate::net::{Context, Descriptor, Layer};

#[derive(Debug, Clone)]
pub struct LogSoftmax {
    descriptor: Descriptor,
}

impl LogSoftmax {
    pub fn new(mut descriptor: Descriptor) -> Self {
        assert_eq!(descriptor.inputs().len(), 1); // Should only be one input.

        descriptor.add_output(descriptor.input(0).unit_shape().clone());

        LogSoftmax { descriptor: descriptor }
    }
}

impl<B: IBackend + conn::LogSoftmax<f32>> Layer<B> for LogSoftmax {
    fn compute_output(&self, backend: &B, context: &mut Context) {
        let input = context.get_data(self.descriptor.input(0));
        let output = context.acquire_data(self.descriptor.output(0));
        backend.log_softmax(&input.borrow(), &mut output.borrow_mut()).unwrap();
    }

    fn compute_gradients(&self, backend: &B, context: &mut Context) {
        let input = context.get_data(self.descriptor.input(0));
        let output = context.get_data(self.descriptor.output(0));
        let output_gradient = context.get_data_gradient(self.descriptor.output(0));
        let input_gradient = context.acquire_data_gradient(self.descriptor.input(0));
        backend
            .log_softmax_grad(
                &output.borrow(),
                &output_gradient.borrow(),
                &mut input_gradient.borrow_mut(),
            )
            .unwrap();
    }

    fn descriptor(&self) -> &Descriptor {
        &self.descriptor
    }

    fn descriptor_mut(&mut self) -> &mut Descriptor {
        &mut self.descriptor
    }
}

#[cfg(test)]
mod tests {
    use coaster::frameworks::native::get_native_backend;

    use crate::net::{layer::testing::*, LayerConfig, Network};

    #[test]
    fn compute() {
        let backend = get_native_backend();
        let net = Network::from_config(&backend, LayerConfig::LogSoftmax, &[vec![2]]).unwrap();
        let result = get_net_output(&backend, &net, &create_tensor_2d([[1.0, -2.0], [3.5, 4.0]]));
        assert_tensor_eq(&result.output, &create_tensor_2d([[-0.04859, -3.04859], [-0.97408, -0.47408]]));
    }

    #[test]
    fn compute_gradients() {
        let backend = get_native_backend();
        let net = Network::from_config(&backend, LayerConfig::LogSoftmax, &[vec![2]]).unwrap();
        let result = get_net_output_and_gradients(
            &backend,
            &net,
            &create_tensor_2d([[1.0, -2.0], [-3.0, 4.0]]),
            &create_tensor_2d([[0.4, 0.3], [0.1, 0.2]]),
        );
        assert_tensor_eq(&result.input_gradient, &create_tensor_2d([[-0.55257, 0.25257], [0.09909, -0.79909]]));
    }
}
