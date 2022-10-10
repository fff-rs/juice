use crate::co::{IBackend, ITensorDesc};
use crate::coblas::plugin::*;
use crate::net::{Context, Descriptor, Layer};
use crate::util::{native_backend, Axpby};

#[derive(Debug)]
// Layer implementing the Mean Squared Error loss function.
// This implementation supports sparse labels (marked as NaN values in label tensor).
// Gradient for absent label values will be 0.0.
pub struct MeanSquaredError {
    descriptor: Descriptor,
}

impl MeanSquaredError {
    pub fn new(descriptor: Descriptor) -> Self {
        assert_eq!(
            descriptor.inputs().len(),
            2,
            "MeanSquaredError must take 2 inputs: values and labels"
        );
        assert_eq!(
            descriptor.inputs()[0].unit_shape().size(),
            descriptor.inputs()[1].unit_shape().size(),
            "Labels must be of the same size"
        );
        // Loss layers don't have outputs.

        MeanSquaredError {
            descriptor: descriptor,
        }
    }
}

impl<B: IBackend + Axpby<f32> + Copy<f32>> Layer<B> for MeanSquaredError {
    fn compute_output(&self, backend: &B, context: &mut Context) {
        // No output computation since loss layer doesn't have outputs.
        // It's main purpose is to start the backpropagation process by
        // computing the loss gradient with respect to net final output
        // in compute_gradients().
    }

    fn compute_gradients(&self, backend: &B, context: &mut Context) {
        let predictions = context.get_data(self.descriptor.input(0));
        let labels = context.get_data(self.descriptor.input(1));
        let input_gradient = context.acquire_data_gradient(self.descriptor.input(0));

        let native = native_backend();

        let predictions_ref = predictions.borrow();
        let predictions_data = predictions_ref
            .read(native.device())
            .unwrap()
            .as_slice::<f32>();

        let labels_ref = labels.borrow();
        let labels_data = labels_ref.read(native.device()).unwrap().as_slice::<f32>();

        let mut input_gradient_ref = input_gradient.borrow_mut();
        let input_gradient_data = input_gradient_ref
            .write_only(native.device())
            .unwrap()
            .as_mut_slice::<f32>();

        for i in 0..predictions_data.len() {
            input_gradient_data[i] = match labels_data[i].is_nan() {
                true => 0.0,
                false => 2.0 * (predictions_data[i] - labels_data[i]),
            };
        }
    }

    fn descriptor(&self) -> &Descriptor {
        &self.descriptor
    }

    fn descriptor_mut(&mut self) -> &mut Descriptor {
        &mut self.descriptor
    }
}
