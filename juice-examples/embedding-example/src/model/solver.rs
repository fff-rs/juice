use crate::model::params::{EMBED_DIMENSION, PHRASE_LENGTH, VOCAB_SIZE, OUTPUT_SIZE};
use coaster::frameworks::cuda::get_cuda_backend;
use coaster::prelude::*;
use coaster_nn::{DirectionMode, RnnInputMode, RnnNetworkMode};
use juice::layer::*;
use juice::layers::*;
use juice::solver::*;
use juice::util::*;
use std::rc::Rc;

pub(crate) fn add_solver(
    net_cfg: SequentialConfig,
    backend: Rc<Backend<Cuda>>,
    batch_size: usize,
    learning_rate: f32,
    momentum: f32,
) -> Solver<Backend<Cuda>, Backend<Cuda>> {
    // Define an Objective Function
    let mut classifier_cfg = SequentialConfig::default();

    classifier_cfg.add_input("data_output", &[batch_size, OUTPUT_SIZE]);
    classifier_cfg.add_input("label", &[batch_size, 1]);

    // Add a Layer expressing Mean Squared Error (MSE) Loss. This will be used with the solver to
    // train the model.
    let nll_layer_cfg = NegativeLogLikelihoodConfig { num_classes: OUTPUT_SIZE };
    let nll_cfg = LayerConfig::new("nll", LayerType::NegativeLogLikelihood(nll_layer_cfg));
    classifier_cfg.add_layer(nll_cfg);

    // Setup an Optimiser
    let mut solver_cfg = SolverConfig {
        minibatch_size: batch_size,
        base_lr: learning_rate,
        momentum,
        ..SolverConfig::default()
    };

    solver_cfg.network = LayerConfig::new("network", net_cfg);
    solver_cfg.objective = LayerConfig::new("classifier", classifier_cfg);
    Solver::from_config(backend.clone(), backend, &solver_cfg)
}
