use std::fmt::{Debug, Formatter};

use crate::co::IBackend;
use crate::net::{Context, Descriptor, Layer};
use crate::util::LayerOps;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DropoutConfig {
    /// The probability to clamp a value to zero.
    pub probability: f32,
    /// The initial seed for the (pseudo-)random generator.
    pub seed: u64,
}

pub struct Dropout<B: conn::Dropout<f32>> {
    descriptor: Descriptor,

    // Backend-specific dropout config.
    backend_config: B::CDROP,
}

impl DropoutConfig {
    pub fn new(probability: f32, seed: u64) -> Self {
        DropoutConfig { probability, seed }
    }
}

impl<B: conn::Dropout<f32>> Dropout<B> {
    pub fn new(backend: &B, mut descriptor: Descriptor, config: &DropoutConfig) -> Self {
        assert_eq!(descriptor.inputs().len(), 1); // Should be only one input.

        descriptor.add_output(descriptor.input(0).unit_shape().clone());

        let backend_config = backend.new_dropout_config(config.probability, config.seed).unwrap();

        Dropout {
            descriptor,
            backend_config,
        }
    }
}

impl<B: IBackend + LayerOps<f32>> Layer<B> for Dropout<B> {
    fn compute_output(&self, backend: &B, context: &mut Context) {
        let input = context.get_data(self.descriptor.input(0));
        let output = context.acquire_data(self.descriptor.output(0));

        backend
            .dropout(&input.borrow(), &mut output.borrow_mut(), &self.backend_config)
            .unwrap();
    }

    fn compute_gradients(&self, backend: &B, context: &mut Context) {
        let input = context.get_data(self.descriptor.input(0));
        let output = context.get_data(self.descriptor.output(0));
        let output_gradient = context.get_data_gradient(self.descriptor.output(0));

        let input_gradient = context.acquire_data_gradient(self.descriptor.input(0));

        backend
            .dropout_grad(
                &output.borrow(),
                &output_gradient.borrow(),
                &input.borrow(),
                &mut input_gradient.borrow_mut(),
                &self.backend_config,
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

impl<B: conn::Dropout<f32>> Debug for Dropout<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dropout").field("descriptor", &self.descriptor).finish()
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use coaster::frameworks::native::get_native_backend;

    use crate::net::{testing::*, Network};

    use super::DropoutConfig;

    #[test]
    fn compute() {
        let backend = get_native_backend();
        let net = Network::from_config(&backend, DropoutConfig::new(0.1, 1), &[vec![2]]).unwrap();
        let result = get_net_output(&backend, &net, &create_tensor_2d([[1.0, 2.0], [3.0, 4.0]]));
        assert_tensor_eq(&result.output, &create_tensor_2d([[1.0, 0.0], [3.0, 4.0]]));
    }

    // TODO: dropout_grad() is not implemented for native backend. Either implement it
    // or convert the tests to CUDA backend (the latter is currently also broken).
}
