use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::coblas::plugin::Copy;
use crate::train::Optimizer;
use crate::util::{native_scalar, Axpby};
use crate::weight::FillerType;
use co::prelude::*;

#[derive(Clone, Debug)]
pub struct SgdWithMomentumConfig {
    pub momentum: f32,
}

// SGD with momentum.
// Computes the update Vᵢ from params gradient ∇ᵢ as:
//   Vᵢ=(1-β)Vᵢ₋₁ + β∇ᵢ,
//   V₀ = 0,
// where:
//   β is the momentum parameter (typically set to 0.1).
pub struct SgdWithMomentum {
    history: HashMap<usize, SharedTensor<f32>>,
    // Precomputed tensor constants.
    zero: SharedTensor<f32>,
    momentum: SharedTensor<f32>,
    one_minus_momentum: SharedTensor<f32>,
}

impl Default for SgdWithMomentumConfig {
    fn default() -> Self {
        SgdWithMomentumConfig { momentum: 0.1 }
    }
}

impl SgdWithMomentum {
    pub fn new(config: &SgdWithMomentumConfig) -> Self {
        SgdWithMomentum {
            history: HashMap::new(),
            zero: native_scalar(0.0),
            momentum: native_scalar(config.momentum),
            one_minus_momentum: native_scalar(1.0 - config.momentum),
        }
    }
}

impl<B: IBackend + Axpby<f32> + Copy<f32>> Optimizer<B> for SgdWithMomentum {
    fn adjust_weight_change(
        &mut self,
        backend: &B,
        weight_changes: &HashMap<usize, Rc<RefCell<SharedTensor<f32>>>>,
    ) {
        for (key, change) in weight_changes {
            let mut change_ref = change.borrow_mut();

            let mut history = self.history.entry(*key).or_insert_with(|| {
                let mut tensor = SharedTensor::new(change_ref.desc());
                FillerType::fill_constant(&mut tensor, 0.0);
                tensor
            });

            // Make sure the params shape didn't change under us.
            assert_eq!(history.desc().size(), change_ref.desc().size());

            // Compute Vᵢ=(1-β)Vᵢ₋₁ + β∇.
            backend
                .axpby(
                    &self.momentum,
                    &change_ref,
                    &self.one_minus_momentum,
                    history,
                )
                .unwrap();

            // Copy Vᵢ to the weight change which should hold the return value.
            backend.copy(history, &mut change_ref).unwrap();
        }
    }
}
