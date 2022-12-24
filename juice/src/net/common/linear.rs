use std::cell::RefCell;
use std::rc::Rc;

use crate::co::{IBackend, ITensorDesc, SharedTensor};
use crate::coblas::transpose::Transpose;
use crate::net::{Context, Descriptor, Layer, LearnableParams};
use crate::util::{native_scalar, LayerOps};
use crate::weight::FillerType;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LinearConfig {
    pub output_size: usize,
}

#[derive(Debug)]
pub struct Linear {
    descriptor: Descriptor,

    // Weight (A) and bias (b) for the linear operation y = Ax + b.
    weight: Rc<RefCell<LearnableParams>>,
    bias: Rc<RefCell<LearnableParams>>,

    // Constants saved for efficiency.
    one: SharedTensor<f32>,
    zero: SharedTensor<f32>,
}

impl LinearConfig {
    pub fn new(output_size: usize) -> Self {
        LinearConfig { output_size }
    }
}

impl Linear {
    pub fn new(mut descriptor: Descriptor, config: &LinearConfig) -> Self {
        assert_eq!(descriptor.inputs().len(), 1); // Should be only one input.
        let input_size = descriptor.input(0).unit_shape().size();

        descriptor.add_output(vec![config.output_size]);

        // Create weight matrix.
        // TODO: Explain why weights are transposed.
        let mut weight = SharedTensor::<f32>::new(&[config.output_size, input_size]);
        FillerType::fill_glorot(&mut weight, input_size, config.output_size);

        // Create bias and initialize with 0. (See https://cs231n.github.io/neural-networks-2/#init for
        // some discussion on proper bias init values.)
        let mut bias = SharedTensor::<f32>::new(&[1, config.output_size]);
        FillerType::fill_constant(&mut bias, 0.0);

        let weight_param = descriptor.create_params("weights", weight, 1.0);
        let bias_param = descriptor.create_params("bias", bias, 1.0);

        Linear {
            descriptor: descriptor,
            weight: weight_param,
            bias: bias_param,
            one: native_scalar(1f32),
            zero: native_scalar(0f32),
        }
    }
}

impl<B: IBackend + LayerOps<f32>> Layer<B> for Linear {
    fn compute_output(&self, backend: &B, context: &mut Context) {
        let input = context.get_data(self.descriptor.input(0));
        let output = context.acquire_data(self.descriptor.output(0));

        let mut ones_tensor = SharedTensor::<f32>::new(&[context.batch_size(), 1]);
        FillerType::fill_constant(&mut ones_tensor, 1f32);

        backend
            .gemm(
                &self.one,
                Transpose::NoTrans,
                &ones_tensor,
                Transpose::NoTrans,
                &self.bias.borrow().data,
                &self.zero,
                &mut output.borrow_mut(),
            )
            .unwrap();

        backend
            .gemm(
                &self.one,
                Transpose::NoTrans,
                &input.borrow(),
                Transpose::Trans,
                &self.weight.borrow().data,
                &self.one,
                &mut output.borrow_mut(),
            )
            .unwrap();
    }

    fn compute_gradients(&self, backend: &B, context: &mut Context) {
        let input = context.get_data(self.descriptor.input(0));
        let output_gradient = context.get_data_gradient(self.descriptor.output(0));

        let input_gradient = context.acquire_data_gradient(self.descriptor.input(0));
        let weights_gradient = context.acquire_params_gradient(self.descriptor.param(0));
        let bias_gradient = context.acquire_params_gradient(self.descriptor.param(1));

        // Network error gradient with respect to input data.
        // dE/dx = dE/dy * df/dx = dE/dy * w.
        backend
            .gemm(
                &self.one,
                Transpose::NoTrans,
                &output_gradient.borrow(),
                Transpose::NoTrans,
                &self.weight.borrow().data,
                &self.zero,
                &mut input_gradient.borrow_mut(),
            )
            .unwrap();

        // Network error gradient with respect to weights.
        // dE/dw = dE/dy * df/dw = dE/dy * x.
        backend
            .gemm(
                &self.one,
                Transpose::Trans,
                &output_gradient.borrow(),
                Transpose::NoTrans,
                &input.borrow(),
                &self.zero,
                &mut weights_gradient.borrow_mut(),
            )
            .unwrap();

        // Network error gradient with respect to bias.
        // dE/dw = dE/dy * df/db = dE/dy * [1] = dE/dy.
        let mut ones_row = SharedTensor::new(&vec![1, context.batch_size()]);
        FillerType::fill_constant(&mut ones_row, 1.0);
        backend
            .gemm(
                &self.one,
                Transpose::NoTrans,
                &ones_row,
                Transpose::NoTrans,
                &output_gradient.borrow(),
                &self.zero,
                &mut bias_gradient.borrow_mut(),
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

    use super::LinearConfig;

    #[test]
    fn compute() {
        let backend = get_native_backend();
        let net = Network::from_config(
            &backend,
            LayerConfig::Linear(LinearConfig { output_size: 2 }),
            &[vec![3]],
        )
        .unwrap();

        // Set params such that layer becomes this:
        //            |1 4|
        // |x1 x2 x3| |2 5| + |0.1 0.2|
        //            |3 6|
        // Note that weights are stored transposed.
        set_params(&net.top().descriptor().params()[0], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        set_params(&net.top().descriptor().params()[1], &[0.1, 0.2]);

        let result = get_net_output(&backend, &net, &create_tensor_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));

        assert_tensor_eq(&result.output, &create_tensor_2d([[14.1, 32.2], [32.1, 77.2]]));
    }

    #[test]
    fn compute_gradients() {
        let backend = get_native_backend();
        let net = Network::from_config(
            &backend,
            LayerConfig::Linear(LinearConfig { output_size: 2 }),
            &[vec![3]],
        )
        .unwrap();

        // Set params such that layer becomes this:
        //            |1 4|
        // |x1 x2 x3| |2 5| + |0.1 0.2|
        //            |3 6|
        // Note that weights are stored transposed.
        set_params(&net.top().descriptor().params()[0], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        set_params(&net.top().descriptor().params()[1], &[0.1, 0.2]);

        // Output gradient contains a single non-zero item at pos 0,0.
        {
            let result = get_net_output_and_gradients(
                &backend,
                &net,
                &create_tensor_2d([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]),
                &create_tensor_2d([[1.0, 0.0], [0.0, 0.0]]),
            );
            assert_tensor_eq(
                &result.input_gradient,
                &create_tensor_2d([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]),
            );
            assert_eq!(result.params_gradients.len(), 2);
            assert_tensor_eq(
                &result.params_gradients[0],
                &create_tensor_2d([[0.01, 0.02, 0.03], [0.0, 0.0, 0.0]]),
            );
            assert_tensor_eq(&result.params_gradients[1], &create_tensor_2d([[1.0, 0.0]]));
        }

        // Output gradient contains a single non-zero item at pos 0,1.
        {
            let result = get_net_output_and_gradients(
                &backend,
                &net,
                &create_tensor_2d([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]),
                &create_tensor_2d([[0.0, 1.0], [0.0, 0.0]]),
            );
            assert_tensor_eq(
                &result.input_gradient,
                &create_tensor_2d([[4.0, 5.0, 6.0], [0.0, 0.0, 0.0]]),
            );
            assert_eq!(result.params_gradients.len(), 2);
            assert_tensor_eq(
                &result.params_gradients[0],
                &create_tensor_2d([[0.0, 0.0, 0.0], [0.01, 0.02, 0.03]]),
            );
            assert_tensor_eq(&result.params_gradients[1], &create_tensor_2d([[0.0, 1.0]]));
        }

        // Output gradient contains a single non-zero item at pos 1,0.
        {
            let result = get_net_output_and_gradients(
                &backend,
                &net,
                &create_tensor_2d([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]),
                &create_tensor_2d([[0.0, 0.0], [1.0, 0.0]]),
            );
            assert_tensor_eq(
                &result.input_gradient,
                &create_tensor_2d([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
            );
            assert_eq!(result.params_gradients.len(), 2);
            assert_tensor_eq(
                &result.params_gradients[0],
                &create_tensor_2d([[0.04, 0.05, 0.06], [0.0, 0.0, 0.0]]),
            );
            assert_tensor_eq(&result.params_gradients[1], &create_tensor_2d([[1.0, 0.0]]));
        }

        // Output gradient contains a single non-zero item at pos 1,1.
        {
            let result = get_net_output_and_gradients(
                &backend,
                &net,
                &create_tensor_2d([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]),
                &create_tensor_2d([[0.0, 0.0], [0.0, 1.0]]),
            );
            assert_tensor_eq(
                &result.input_gradient,
                &create_tensor_2d([[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]]),
            );
            assert_eq!(result.params_gradients.len(), 2);
            assert_tensor_eq(
                &result.params_gradients[0],
                &create_tensor_2d([[0.0, 0.0, 0.0], [0.04, 0.05, 0.06]]),
            );
            assert_tensor_eq(&result.params_gradients[1], &create_tensor_2d([[0.0, 1.0]]));
        }

        // Output gradient contains all 1s.
        {
            let result = get_net_output_and_gradients(
                &backend,
                &net,
                &create_tensor_2d([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]),
                &create_tensor_2d([[1.0, 1.0], [1.0, 1.0]]),
            );
            assert_tensor_eq(
                &result.input_gradient,
                &create_tensor_2d([[5.0, 7.0, 9.0], [5.0, 7.0, 9.0]]),
            );
            assert_eq!(result.params_gradients.len(), 2);
            assert_tensor_eq(
                &result.params_gradients[0],
                &create_tensor_2d([[0.05, 0.07, 0.09], [0.05, 0.07, 0.09]]),
            );
            assert_tensor_eq(&result.params_gradients[1], &create_tensor_2d([[2.0, 2.0]]));
        }
    }
}
