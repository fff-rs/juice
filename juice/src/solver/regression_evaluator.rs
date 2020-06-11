//! Set of Evaluators for Regression Problems

use crate::co::SharedTensor;
use crate::util::native_backend;
use std::collections::VecDeque;
use std::fmt;
/// Sampled Evaluator for Regression Problems
///
/// Produces measure of accuracy for regression problems up to `Capacity` elements in a
/// First-In First-Out stack.
#[derive(Debug)]
pub struct RegressionEvaluator {
    evaluation_metric: String,
    capacity: Option<usize>,
    samples: VecDeque<Sample>,
}

impl RegressionEvaluator {
    /// Create an evaluator for Regression Problems
    ///
    /// # Arguments
    /// * `evaluation_metric` - Regression metric to use for evaluation - i.e. 'mse'
    pub fn new(evaluation_metric: Option<String>) -> RegressionEvaluator {
        RegressionEvaluator {
            evaluation_metric: evaluation_metric.unwrap_or("mse".to_string()),
            capacity: None,
            samples: VecDeque::new(),
        }
    }

    /// Add a sample by providing the expected `target` value and the `prediction`.
    pub fn add_sample(&mut self, prediction: f32, target: f32) {
        if self.capacity.is_some() && self.samples.len() >= self.capacity.unwrap() {
            self.samples.pop_front();
        }
        self.samples.push_back(Sample { prediction, target });
    }

    /// Add a batch of samples.
    ///
    /// See [add_sample](#method.add_sample).
    pub fn add_samples(&mut self, predictions: &[f32], targets: &[f32]) {
        for (&prediction, &target) in predictions.iter().zip(targets.iter()) {
            self.add_sample(prediction, target)
        }
    }

    /// Get the predicted value from the output of a network.
    pub fn get_predictions(&self, network_out: &mut SharedTensor<f32>) -> Vec<f32> {
        let native_inferred = network_out.read(native_backend().device()).unwrap();
        native_inferred.as_slice::<f32>().to_vec()
    }

    /// Set the `capacity` of the Regression Evaluator
    pub fn set_capacity(&mut self, capacity: Option<usize>) {
        self.capacity = capacity;
        // TODO: truncate if over capacity
    }

    /// Return all collected samples.
    pub fn samples(&self) -> &VecDeque<Sample> {
        &self.samples
    }

    /// Return the accuracy of the collected predictions.
    pub fn accuracy(&self) -> impl RegressionLoss {
        let num_samples = self.samples.len();
        match &*self.evaluation_metric {
            "mse" => {
                let sum_squared_error = self
                    .samples
                    .iter()
                    .fold(0.0, |acc, sample| acc + (sample.prediction - sample.target).powi(2));
                MeanSquaredErrorAccuracy {
                    num_samples,
                    sum_squared_error,
                }
            }
            _ => unimplemented!(),
        }
    }
}

/// A single prediction sample.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    prediction: f32,
    target: f32,
}

impl fmt::Display for Sample {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Prediction: {:.2?}, Target: {:.2?}", self.prediction, self.target)
    }
}

/// Trait to show loss & metric for a Regression Evaluator
pub trait RegressionLoss {
    /// Loss function to produce metric
    fn loss(&self) -> f32;
}

impl RegressionLoss for MeanSquaredErrorAccuracy {
    fn loss(&self) -> f32 {
        self.sum_squared_error / self.num_samples as f32
    }
}

#[derive(Debug, Clone, Copy)]
/// Provides loss calculated by Mean Squared Error for sampled data
///
/// Calculated as 1/N Σ (Prediction - Actual)^2 where N is the number of samples.
pub struct MeanSquaredErrorAccuracy {
    num_samples: usize,
    sum_squared_error: f32,
}

#[allow(trivial_casts)]
impl fmt::Display for dyn RegressionLoss {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, " {:.6?}", self.loss())
    }
}
