use coaster::ITensorDesc;

use crate::co::IBackend;
use crate::conn;
use crate::net::{Context, Descriptor, Layer};

#[derive(Debug, Clone)]
pub struct Softmax {
    descriptor: Descriptor,
}

impl Softmax {
    pub fn new(mut descriptor: Descriptor) -> Self {
        assert_eq!(descriptor.inputs().len(), 1); // Should only be one input.

        descriptor.add_output(descriptor.input(0).unit_shape().clone());

        Softmax { descriptor: descriptor }
    }
}

impl<B: IBackend + conn::Softmax<f32>> Layer<B> for Softmax {
    fn compute_output(&self, backend: &B, context: &mut Context) {
        let input = context.get_data(self.descriptor.input(0));
        let output = context.acquire_data(self.descriptor.output(0));

        // Since the backend expects the first dimension to be the batch number,
        // verify that this assumption holds.
        let batch_item_size = input.borrow().desc().iter().skip(1).fold(1, |acc, v| acc * v);
        assert_eq!(batch_item_size, self.descriptor.input(0).unit_shape().size());

        backend.softmax(&input.borrow(), &mut output.borrow_mut()).unwrap();
    }

    fn compute_gradients(&self, backend: &B, context: &mut Context) {
        let input = context.get_data(self.descriptor.input(0));
        let output = context.get_data(self.descriptor.output(0));
        let output_gradient = context.get_data_gradient(self.descriptor.output(0));
        let input_gradient = context.acquire_data_gradient(self.descriptor.input(0));
        backend
            .softmax_grad(
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
        let net = Network::from_config(&backend, LayerConfig::Softmax, &[vec![2]]).unwrap();
        let result = get_net_output(&backend, &net, &create_tensor_2d([[1.0, -2.0], [3.5, 4.0]]));
        assert_tensor_eq(
            &result.output,
            &create_tensor_2d([[0.95257, 0.04743], [0.37754, 0.62246]]),
        );
    }

    #[test]
    fn compute_gradients() {
        let backend = get_native_backend();
        let net = Network::from_config(&backend, LayerConfig::Softmax, &[vec![2]]).unwrap();
        let result = get_net_output_and_gradients(
            &backend,
            &net,
            &create_tensor_2d([[1.0, -2.0], [3.5, 4.0]]),
            &create_tensor_2d([[0.4, 0.3], [0.1, 0.2]]),
        );
        assert_tensor_eq(
            &result.input_gradient,
            &create_tensor_2d([[-0.15003, -0.01221], [-0.17273, -0.22253]]),
        );
    }
}
