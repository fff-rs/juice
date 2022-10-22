use crate::co::IBackend;
use crate::conn;
use crate::net::{Context, Descriptor, Layer};

#[derive(Debug, Clone)]
pub struct Sigmoid {
    descriptor: Descriptor,
}

impl Sigmoid {
    pub fn new(mut descriptor: Descriptor) -> Self {
        assert_eq!(descriptor.inputs().len(), 1); // Should only be one input.

        descriptor.add_output(descriptor.input(0).unit_shape().clone());

        Sigmoid { descriptor }
    }
}

impl<B: IBackend + conn::Sigmoid<f32>> Layer<B> for Sigmoid {
    fn compute_output(&self, backend: &B, context: &mut Context) {
        let input = context.get_data(self.descriptor.input(0));
        let output = context.acquire_data(self.descriptor.output(0));
        backend.sigmoid(&input.borrow(), &mut output.borrow_mut()).unwrap();
    }

    fn compute_gradients(&self, backend: &B, context: &mut Context) {
        let input = context.get_data(self.descriptor.input(0));
        let output = context.get_data(self.descriptor.output(0));
        let output_gradient = context.get_data_gradient(self.descriptor.output(0));
        let input_gradient = context.acquire_data_gradient(self.descriptor.input(0));
        backend
            .sigmoid_grad(
                &output.borrow(),
                &output_gradient.borrow(),
                &input.borrow(),
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
    use crate::net::{layer::testing::*, LayerConfig, Network};

    #[test]
    fn compute() {
        let net = Network::from_config(LayerConfig::Sigmoid, &[vec![2]]).unwrap();
        let result = get_net_output(&net, &[[1.0, -2.0], [-3.0, 4.0]]);
        assert_tensor_eq(&result.output, &[[0.7310586, 0.11920292], [0.047425874, 0.98201376]]);
    }

    #[test]
    fn compute_gradients() {
        let net = Network::from_config(LayerConfig::Sigmoid, &[vec![2]]).unwrap();
        let result = get_net_output_and_gradients(&net, &[[1.0, -2.0], [-3.0, 4.0]], &[[0.4, 0.3], [0.1, 0.2]]);
        assert_tensor_eq(
            &result.input_gradient,
            &[[0.078644775, 0.031498075], [0.004517666, 0.0035325468]],
        );
    }
}
