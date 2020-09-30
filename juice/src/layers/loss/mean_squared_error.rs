//! Provides Loss & Gradient for Mean Squared Error
//!
//! Calculation of [Mean Squared Error][1] for regression problems.
//!
//! [1]: https://en.wikipedia.org/wiki/Mean_squared_error

use crate::co::prelude::*;
use crate::layer::*;
use crate::util::*;

#[derive(Debug, Clone)]
#[allow(missing_copy_implementations)]
/// Mean Squared Error Layer
pub struct MeanSquaredError;

impl MeanSquaredError {
    fn batch_size(input_shape: &[usize]) -> usize {
        match input_shape.len() {
            1 => 1,
            2 => input_shape[0],
            _ => panic!("MSE layer only supports 1D/2D inputs"),
        }
    }
}

impl<B: IBackend + LayerOps<<B as IBackend>::F,f32> + Axpby<f32>> ILayer<B> for MeanSquaredError {
    fn reshape(
        &mut self,
        backend: ::std::rc::Rc<B>,
        input_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
        input_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
        weights_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
        weights_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
        output_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
        output_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
    ) {
        let input_desc = input_data[0].read().unwrap().desc().clone();
        input_gradient[0].write().unwrap().resize(&input_desc).unwrap();
        output_data[0].write().unwrap().resize(&input_desc).unwrap();
        output_gradient[0].write().unwrap().resize(&input_desc).unwrap();
    }
}

// Calculate Loss an as Output
impl<B: IBackend> ComputeOutput<f32, B> for MeanSquaredError {
    fn compute_output(
        &self,
        backend: &B,
        _weights: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        output_data: &mut [&mut SharedTensor<f32>],
    ) {
        let predictions = input_data[0];
        let labels = input_data[1];
        let batch_size = Self::batch_size(labels.desc());
        let native = native_backend();
        let native_labels = labels.read(native.device()).unwrap().as_slice::<f32>();
        let native_predictions = predictions.read(native.device()).unwrap().as_slice::<f32>();

        let writable_loss = native_labels
            .iter()
            .zip(native_predictions)
            .fold(0f32, |acc, (label, prediction)| acc + (prediction - label).powi(2));

        write_to_memory(
            output_data[0].write_only(native.device()).unwrap(),
            &vec![writable_loss / batch_size as f32],
        );
    }
}

// Calculate a Gradient for Mean Squared Error
impl<B: IBackend + LayerOps<<B as IBackend>::F,f32>> ComputeInputGradient<f32, B> for MeanSquaredError {
    fn compute_input_gradient(
        &self,
        backend: &B,
        weights_data: &[&SharedTensor<f32>],
        output_data: &[&SharedTensor<f32>],
        output_gradients: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        input_gradients: &mut [&mut SharedTensor<f32>],
    ) {
        let labels = input_data[1];
        let predictions = input_data[0];
        let batch_size = Self::batch_size(labels.desc());
        let native = native_backend();
        let native_labels = labels.read(native.device()).unwrap().as_slice::<f32>();
        let native_predictions = predictions.read(native.device()).unwrap().as_slice::<f32>();

        let mut writable_input: SharedTensor<f32> = SharedTensor::new(labels.desc());
        write_to_memory(writable_input.write_only(native.device()).unwrap(), native_labels);

        // Gradient is calculated as 2 * (Predictions - Labels)
        Axpby::axpby(
            backend,
            &native_scalar(2f32),
            &predictions,
            &native_scalar(-2f32),
            &mut writable_input,
        )
            .unwrap();

        write_to_memory(
            input_gradients[0].write_only(native.device()).unwrap(),
            &writable_input.read(native.device()).unwrap().as_slice::<f32>(),
        );
    }
}

impl<B: IBackend> ComputeParametersGradient<f32, B> for MeanSquaredError {}

impl ::std::default::Default for MeanSquaredError {
    fn default() -> MeanSquaredError {
        MeanSquaredError
    }
}
