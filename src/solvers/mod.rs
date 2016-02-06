//! Provides the trainers for the [Network][network].
//! [network]: ../network/index.html
//!
//! The optimal state of a neural network would be the one where
//! for any given input to the network, it would produce an output perfectly
//! matching the target function. In that state the loss function would have its
//! [global minimum][minimum].
//! This statement can also be reversed to *if we manage to minimize
//! the loss function of the network, we map the target function*.
//!
//! We can change the way a network works by adjusting its individual
//! [weights][weight]. So to optimize the network we want to adjust
//! the weights in a way that the loss function will be minimized.
//! If we want to know how to correctly adjust a single weight,
//! we have to get to know the effect of that weight
//! on the loss function (= the *gradient*).
//! This can be done via a method called [*backpropagation*][backprop].
//!
//! There are different methods of how a Solver solves for the minimum of the
//! loss function. They mostly differ in two ways:
//!
//! - How to execute the backpropagation to compute the gradient.
//! - How to comute the weight update from the gradient.
//!
//! [loss]: ../layers/loss/index.html
//! [weight]: https://en.wikipedia.org/wiki/Synaptic_weight
//! [minimum]: http://mathworld.wolfram.com/GlobalMinimum.html
//! [backprop]: https://en.wikipedia.org/wiki/Backpropagation

#[allow(unused_import_braces)]
pub use self::sgd::{Momentum};
pub mod sgd;

use co::backend::IBackend;
use co::memory::MemoryType;
use co::tensor::SharedTensor;
// use coblas::plugin::IBlas;
use conn::NN;
use shared_memory::*;
use solver::*;
use network::Network;
use util::{native_backend, LayerOps, SolverOps};

trait SGDSolver<SolverB: IBackend + SolverOps<f32>, NetB: IBackend + LayerOps<f32>> : ISolver<SolverB, NetB> {
    fn compute_update_value(&mut self,
                            config: &SolverConfig,
                            weight_blob: &ArcLock<HeapBlob>,
                            history_blob_id: usize,
                            global_lr: &f32,
                            blob_lr: &f32);

    /// [Clip gradients][1] when they exceed [SolverConfig.clip_gradients][2].
    /// [1]: http://arxiv.org/abs/1211.5063
    /// [2]: ../solver/struct.SolverConfig.html
    ///
    /// [Gradient norm clipping][1] is a technique used when dealing with
    /// [Recurrent Neural Networks][3].
    /// When the [L2 norm][4] of the gradients exceeds a threshold it is "clipped"
    /// to that threshold. The naming can be misleading since the gradients are not
    /// actually clipped (as in cut off), but rescaled to the threshold.
    ///
    /// [3]: https://en.wikipedia.org/wiki/Recurrent_neural_network
    /// [4]: https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm
    #[allow(unused_must_use)]
    fn clip_gradients<B: IBackend + LayerOps<f32>>(&self, config: &SolverConfig, net: &mut Network<B>) {
        // skip clipping gradients if SolverConfig.clip_gradients is set to None
        if let Some(clip_threshold) = config.clip_gradients {
            let native = native_backend();

            let net_weights = net.learnable_weights();
            let mut sumsq_diff = 0f32;
            let backend = self.backend();
            for weight_blob in net_weights {
                let blob = weight_blob.read().unwrap();
                let mut result = SharedTensor::<f32>::new(backend.device(), &1).unwrap();
                blob.sumsq_diff(self.backend(), &mut result);

                let mut result = SharedTensor::<f32>::new(backend.device(), &1).unwrap();
                match result.add_device(native.device()) { _ => result.sync(native.device()).unwrap() }
                if let &MemoryType::Native(ref sumsq_result) = result.get(native.device()).unwrap() {
                    let sumsq_diff_slice = sumsq_result.as_slice::<f32>();
                    sumsq_diff += sumsq_diff_slice[0];
                } else {
                    panic!();
                }
            }
            let l2norm_diff = sumsq_diff.sqrt();
            if l2norm_diff > clip_threshold {
                let scale_factor = clip_threshold / l2norm_diff;
                info!("Gradient clipping: scaling down gradients (L2 norm {} > {})
                        by scale factor {}",
                      l2norm_diff,
                      clip_threshold,
                      scale_factor);

                let mut scale_shared = SharedTensor::<f32>::new(native.device(), &1).unwrap();
                if let &mut MemoryType::Native(ref mut scale) = scale_shared.get_mut(native.device()).unwrap() {
                    let scale_slice = scale.as_mut_slice::<f32>();
                    scale_slice[0] = scale_factor;
                } else {
                    panic!();
                }

                for weight_blob in net_weights {
                    let mut blob = weight_blob.write().unwrap();
                    let diff = blob.mut_diff();
                    backend.scal(&mut scale_shared, diff);
                }
            }
        }
    }

    /// Scale the gradient to counteract the [SolverConfig.minibatch_size][1]
    /// [1]: ../solver/struct.SolverConfig.html
    ///
    /// To counteract that we are accumulating the gradients over multiple samples,
    /// we need to scale the gradients down to the equivalent of a single sample.</br>
    /// E.g. with a `minibatch_size` of 4 we need to scale the gradient by 0.25 (= 1/4).
    fn normalize(&self, config: &SolverConfig, weight_blob: &ArcLock<HeapBlob>) {
        if config.minibatch_size > 1 {
            let scale_factor = 1f32 / config.minibatch_size as f32;
            let mut write_blob = weight_blob.write().unwrap();
            let native = native_backend();
            let mut scale_factor_shared = SharedTensor::<f32>::new(native.device(), &1).unwrap();
            if let &mut MemoryType::Native(ref mut scale) = scale_factor_shared.get_mut(native.device()).unwrap() {
                let scale_slice = scale.as_mut_slice::<f32>();
                scale_slice[0] = scale_factor;
            } else {
                panic!();
            }
            let _ = self.backend().scal_plain(&scale_factor_shared, write_blob.mut_diff());
        }
    }

    /// [Regularize][1] the gradient according to the configured [RegularizationMethod][2].
    /// [1]: https://cs231n.github.io/neural-networks-2/#reg
    /// [2]: ../solver/enum.RegularizationMethod.html
    fn regularize(&self, config: &SolverConfig, weight_blob: &ArcLock<HeapBlob>, blob_weight_decay: Option<f32>) {
        if let Some(global_weight_decay) = config.weight_decay {
            if let Some(regularization_method) = config.regularization_method {
                match blob_weight_decay {
                    Some(weight_decay_mult) => {
                        let local_decay = global_weight_decay * weight_decay_mult;
                        match regularization_method {
                            RegularizationMethod::L2 => {
                                let native = native_backend();
                                let mut decay_shared = SharedTensor::<f32>::new(native.device(), &1).unwrap();
                                if let &mut MemoryType::Native(ref mut decay) = decay_shared.get_mut(native.device()).unwrap() {
                                    let decay_slice = decay.as_mut_slice::<f32>();
                                    decay_slice[0] = local_decay;
                                } else {
                                    panic!();
                                }
                                let blob = &mut weight_blob.write().unwrap();
                                blob.regularize_l2(self.backend(), &decay_shared);
                            }
                        }
                    }
                    None => {
                        error!("Weight decay multiplier for blob missing.");
                    }
                }
            }
        }
    }
}
