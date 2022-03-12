//! Adam optimizer.
//! Computes the update Vᵢ from params gradient ∇ᵢ as:
//!   Mᵢ = β₁Mᵢ₋₁ + (1-β₁)∇ᵢ,
//!   Sᵢ = β₂Sᵢ₋₁ + (1-β₂)∇ᵢ⊙∇ᵢ,
//!   M₀ = 0,
//!   S₀ = 0,
//!   M̂ᵢ = Mᵢ/(1-β₁ᵗ),
//!   Ŝᵢ = Sᵢ/(1-β₂ᵗ),
//!   Vᵢ = M̂ᵢ⊘(√Ŝᵢ+ε),
//! where:
//!   ⊙ - pointwise multiplication,
//!   ⊘ - pointwise division,
//!   β₁, β₂ - averaging parameters (typically set to 0.9 and 0.999 respectively),
//!   ε - small constant to prevent division by zero (typically 1e-8).
//!
//! (Note that the update Vᵢ is then additionally scaled by Trainer using global and param-specific
//! learning rates.)

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::coblas::plugin::Copy;
use crate::train::Optimizer;
use crate::util::native_backend;
use crate::weight::FillerType;
use co::prelude::*;

#[derive(Clone, Debug)]
pub struct AdamConfig {
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
}

pub struct Adam {
    // First gradient moment (Mᵢ).
    first_moments: HashMap<usize, SharedTensor<f32>>,
    // Second gradient moment (Sᵢ).
    second_moments: HashMap<usize, SharedTensor<f32>>,

    // Original β₁ as well as raised to t-th power (β₁ᵗ).
    beta1: f32,
    beta1_nth: f32,
    // Original β₂ as well as raised to t-th power (β₂ᵗ).
    beta2: f32,
    beta2_nth: f32,

    epsilon: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        AdamConfig {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
        }
    }
}

impl Adam {
    pub fn new(config: &AdamConfig) -> Self {
        Adam {
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            beta1: config.beta1,
            beta1_nth: config.beta1,
            beta2: config.beta2,
            beta2_nth: config.beta2,
            epsilon: config.epsilon,
        }
    }
}

// TODO: Rewrite with backend ops (requires element-wise square and square root support).
impl<B: IBackend> Optimizer<B> for Adam {
    fn adjust_weight_change(
        &mut self,
        backend: &B,
        weight_changes: &HashMap<usize, Rc<RefCell<SharedTensor<f32>>>>,
    ) {
        let native = native_backend();

        for (key, change) in weight_changes {
            let mut change_ref = change.borrow_mut();

            let mut first_moment = self.first_moments.entry(*key).or_insert_with(|| {
                let mut tensor = SharedTensor::new(change_ref.desc());
                FillerType::fill_constant(&mut tensor, 0.0);
                tensor
            });
            let mut second_moment = self.second_moments.entry(*key).or_insert_with(|| {
                let mut tensor = SharedTensor::new(change_ref.desc());
                FillerType::fill_constant(&mut tensor, 0.0);
                tensor
            });

            let len = change_ref.desc().size();

            let change_slice = change_ref
                .read_write(native.device())
                .unwrap()
                .as_mut_slice::<f32>();
            let first_moment_slice = first_moment
                .read_write(native.device())
                .unwrap()
                .as_mut_slice::<f32>();
            let second_moment_slice = second_moment
                .read_write(native.device())
                .unwrap()
                .as_mut_slice::<f32>();

            // We can rewrite the matrix equations at the top of this file in a element-wise form:
            //   Mᵢ[j] = β₁Mᵢ₋₁[j] + (1-β₁)∇ᵢ[j]
            //   Sᵢ[j] = β₂Sᵢ₋₁[j] + (1-β₂)∇ᵢ[j]²
            //   Vᵢ[j] = Mᵢ[j] / ((1-β₁ᵗ)•√(Sᵢ[j]/(1-β₂ᵗ) + ε)
            for j in 0..len {
                // ∇ᵢ[j].
                let w = change_slice[j];
                // Mᵢ[j], M̂ᵢ[j].
                let m = self.beta1 * first_moment_slice[j] + (1.0 - self.beta1) * w;
                let m_hat = m / (1.0 - self.beta1_nth);
                // Sᵢ[j], Ŝᵢ[j].
                let s = self.beta2 * second_moment_slice[j] + (1.0 - self.beta2) * w * w;
                let s_hat = s / (1.0 - self.beta2_nth);
                // Vᵢ[j].
                let v = m_hat / (s_hat.sqrt() + self.epsilon);
                
                assert!(!v.is_nan());

                change_slice[j] = v;
                first_moment_slice[j] = m;
                second_moment_slice[j] = s;
            }
        }

        self.beta1_nth *= self.beta1;
        self.beta2_nth *= self.beta2;
    }
}
