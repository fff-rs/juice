mod adam;
mod sgd_momentum;

use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;
use std::default::Default;

use crate::coblas::plugin::Copy;
use co::prelude::*;
use crate::util::Axpby;

use adam::Adam;
use sgd_momentum::SgdWithMomentum;

// Expose configs publicly.
pub use adam::AdamConfig;
pub use sgd_momentum::SgdWithMomentumConfig;

// A gradient descent optimization algorithm.
pub trait Optimizer<B: IBackend> {
    // Called on each minibatch training cycle. Takes all weight gradients computed during
    // backpropagation (indexed by an opaque key which is guaranteed to be stable for the
    // duration of the program).
    // Modifies the changes in-place; modified changes will then be applied to the weights:
    //   W = W - α•change,
    // where α is the learning rate (combined from global and param-specific rates).
    fn adjust_weight_change(&mut self, backend: &B, weight_changes: &HashMap<usize, Rc<RefCell<SharedTensor<f32>>>>);
}

#[derive(Clone, Debug)]
pub enum OptimizerConfig {
    SgdWithMomentum(SgdWithMomentumConfig),
    Adam(AdamConfig),
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        OptimizerConfig::SgdWithMomentum(Default::default())
    }
}

pub fn optimizer_from_config<B: IBackend + Axpby<f32> + Copy<f32>>(config: &OptimizerConfig) -> Box<dyn Optimizer<B>> {
    match config {
        OptimizerConfig::SgdWithMomentum(sgd_config) => Box::new(SgdWithMomentum::new(sgd_config)),
        OptimizerConfig::Adam(adam_config) => Box::new(Adam::new(adam_config)),
    }
}

impl From<SgdWithMomentumConfig> for OptimizerConfig {
    fn from(c: SgdWithMomentumConfig) -> Self {
        OptimizerConfig::SgdWithMomentum(c)
    }
}

impl From<AdamConfig> for OptimizerConfig {
    fn from(c: AdamConfig) -> Self {
        OptimizerConfig::Adam(c)
    }
}