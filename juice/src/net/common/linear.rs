use std::cell::RefCell;
use std::rc::Rc;

use crate::co::{IBackend, ITensorDesc, SharedTensor};
use crate::coblas::transpose::Transpose;
use crate::net::{Context, Descriptor, Layer, LearnableParams};
use crate::util::{native_scalar, LayerOps};
use crate::weight::FillerType;

#[derive(Clone, Debug, Default)]
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

impl Linear {
    pub fn new(mut descriptor: Descriptor, config: &LinearConfig) -> Self {
        assert_eq!(descriptor.inputs().len(), 1); // Should be only one input.
        let input_size = descriptor.input(0).unit_shape().size();

        descriptor.add_output(vec![config.output_size]);

        // Create weight matrix.
        let mut weight = SharedTensor::<f32>::new(&[config.output_size, input_size]);
        FillerType::fill_glorot(&mut weight, input_size, config.output_size);
        let mut bias = SharedTensor::<f32>::new(&[config.output_size]);
        FillerType::fill_glorot(&mut weight, input_size, config.output_size);

        // Create bias. Bias is typically intialized with a constant, and a suitable initialisation
        // is stated in https://cs231n.github.io/neural-networks-2/#init for non-LSTM types.
        let mut bias = SharedTensor::<f32>::new(&[config.output_size]);
        let seed = rand::random::<f32>();
        let bias_init_value = seed * (2.0 / seed).sqrt();
        FillerType::fill_constant(&mut bias, bias_init_value);

        let weight_param = descriptor.create_params("weights", weight, 1.0);
        let bias_param = descriptor.create_params("bias", bias, 1.0);

        descriptor.add_params(weight_param.clone());
        descriptor.add_params(bias_param.clone());

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
    fn compute_output(&self, context: &mut Context<B>) {
        let input = context.get_data(self.descriptor.input(0));
        let mut output = context.acquire_data(self.descriptor.output(0));

        let mut ones_tensor = SharedTensor::<f32>::new(&[context.batch_size(), 1]);
        FillerType::fill_constant(&mut ones_tensor, 1f32);

        context
            .backend()
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

        context
            .backend()
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

    fn compute_gradients(&self, context: &mut Context<B>) {
        let input = context.get_data(self.descriptor.input(0));
        let output_gradient = context.get_data_gradient(self.descriptor.output(0));

        let mut input_gradient = context.acquire_data_gradient(self.descriptor.input(0));
        let mut weights_gradient = context.acquire_params_gradient(self.descriptor.param(0));
        let mut bias_gradient = context.acquire_params_gradient(self.descriptor.param(1));

        let mut ones_tensor = SharedTensor::<f32>::new(&[context.batch_size(), 1]);
        FillerType::fill_constant(&mut ones_tensor, 1f32);

        // Network error gradient with respect to input data.
        // dE/dx = dE/dy * df/dx = dE/dy * w.
        context
            .backend()
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
        context
            .backend()
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
        // dE/dw = dE/dy * df/db = dE/dy * [1].
        context
            .backend()
            .gemm(
                &self.one,
                Transpose::Trans,
                &output_gradient.borrow(),
                Transpose::NoTrans,
                &ones_tensor,
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
