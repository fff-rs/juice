use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::co::prelude::*;
use crate::net::*;
use crate::train::{optimizer_from_config, Optimizer, OptimizerConfig, SgdWithMomentumConfig};
use crate::util::{format_tensor, SolverOps};

/// Trainer configuration.
///
/// Specifies minibatch size, objective and optimizer family and parameters.
#[derive(Clone)]
pub struct TrainerConfig {
    pub batch_size: usize,
    pub objective: LayerConfig,
    pub optimizer: OptimizerConfig,
    pub learning_rate: f32,
}

/// Trains a network through minibatch backpropagation.
///
/// Doesn't own the network.
pub struct Trainer<B: IBackend> {
    config: TrainerConfig,

    // Objective (loss) function used for backpropagation.
    objective: Box<dyn Layer<B>>,

    optimizer: Box<dyn Optimizer<B>>,

    iter: usize,
}

fn key_from_rc<T>(rc: &Rc<RefCell<T>>) -> usize {
    Rc::as_ptr(rc) as usize
}

impl Default for TrainerConfig {
    fn default() -> Self {
        TrainerConfig {
            batch_size: 32,
            objective: LayerConfig::MeanSquaredError,
            optimizer: OptimizerConfig::SgdWithMomentum(SgdWithMomentumConfig::default()),
            learning_rate: 0.001,
        }
    }
}

impl<B: IBackend + SolverOps<f32> + 'static> Trainer<B> {
    pub fn from_config(backend: &B, config: TrainerConfig, net: &Network<B>, label_shape: &TensorDesc) -> Self {
        // Create objective.
        let objective_descriptor = Descriptor::top(
            "loss",
            vec![
                net.top().descriptor().output(0).clone(),
                Inout::new(label_shape.clone()),
            ],
        );
        let objective = layer_from_config(backend, objective_descriptor, &config.objective).unwrap();
        let optimizer = optimizer_from_config(&config.optimizer);

        Trainer {
            config,
            objective,
            optimizer,
            iter: 0,
        }
    }

    /// Performs a single minibatch training step.
    /// Returns network output for the provided input.
    pub fn train_minibatch(
        &mut self,
        backend: &B,
        net: &mut Network<B>,
        inputs: &SharedTensor<f32>,
        labels: &SharedTensor<f32>,
    ) -> SharedTensor<f32> {
        trace!("Inputs:\n{}", format_tensor(inputs));
        trace!("Labels:\n{}", format_tensor(labels));

        let batch_size = inputs.desc()[0];
        assert_eq!(batch_size, labels.desc()[0]);

        let mut context = Context::new(batch_size);

        // Copy intput and label data into the context.
        // Copy inputs.
        let context_inputs = context.acquire_data(net.top().descriptor().input(0));
        assert_eq!(context_inputs.borrow().desc().size(), inputs.desc().size());
        backend.copy(&inputs, &mut context_inputs.borrow_mut()).unwrap();
        // Copy labels.
        let context_labels = context.acquire_data(self.objective.descriptor().input(1));
        backend.copy(&labels, &mut context_labels.borrow_mut()).unwrap();

        // Compute network output and the loss.
        net.top().compute_output(backend, &mut context);
        self.objective.compute_output(backend, &mut context);

        trace!(
            "Output:\n{}",
            format_tensor(&context.get_data(net.top().descriptor().output(0)).borrow())
        );

        // Compute params gradients by doing a backpropagation on the network.
        self.objective.compute_gradients(backend, &mut context);
        trace!(
            "Loss gradient:\n{}",
            format_tensor(&context.get_data_gradient(self.objective.descriptor().input(0)).borrow())
        );
        net.top().compute_gradients(backend, &mut context);

        // Collect computed gradients.
        let params_gradients: HashMap<usize, Rc<RefCell<SharedTensor<f32>>>> = net
            .top()
            .descriptor()
            .params()
            .iter()
            .map(|p| {
                (
                    key_from_rc(p),
                    // Use acquire* instead of get* to create missing gradients. This is necessary
                    // because backpropagation might not reach all layers. Created gradients
                    // are zero-filled and thus will not change weights when applied.
                    context.acquire_params_gradient(p),
                )
            })
            .collect();

        // Let the optimizer adjust the params gradients before applying them to params.
        self.optimizer.adjust_weight_change(backend, &params_gradients);

        trace!("Gradient after worker:");
        params_gradients.iter().for_each(|(p, r)| {
            trace!("{:?}: \n{}", p, format_tensor(&r.borrow()));
        });

        // Finally apply the weight change.
        net.top().descriptor().params().iter().for_each(|p| {
            let key = key_from_rc(p);

            let mut params = p.borrow_mut();
            let change = params_gradients.get(&key).unwrap().borrow();

            // When applying the (optimized) gradient, additionally:
            // 1. Normalize for batch size (multiply by 1/batch_size).
            // 2. Multiply by individual params learning rate.
            // 3. Multiply by global learning rate.
            let a = crate::util::native_scalar(
                -params.learning_rate * self.config.learning_rate / (self.config.batch_size as f32),
            );

            backend.axpy(&a, &change, &mut params.data).unwrap();
        });

        self.iter += 1;

        context.take_data(net.top().descriptor().output(0))
    }
}

#[cfg(test)]
mod tests {
    use coaster::{Backend, Native, SharedTensor};
    use rand::Rng;

    use crate::{
        net::{LayerConfig, LinearConfig, Network, SequentialConfig},
        train::{AdamConfig, OptimizerConfig, SgdWithMomentumConfig},
        util::native_backend,
    };

    use super::{Trainer, TrainerConfig};

    const BATCH_SIZE: usize = 32;

    // Creates a batch of inputs and labels representing the sin() function.
    fn create_batch_for_sin(backend: &Backend<Native>) -> (SharedTensor<f32>, SharedTensor<f32>) {
        let mut rng = rand::thread_rng();

        let mut inputs = SharedTensor::new(&vec![BATCH_SIZE, 1]);
        let mut labels = SharedTensor::new(&vec![BATCH_SIZE, 1]);

        let inputs_slice = inputs.write_only(backend.device()).unwrap().as_mut_slice::<f32>();
        let labels_slice = labels.write_only(backend.device()).unwrap().as_mut_slice::<f32>();

        for i in 0..BATCH_SIZE {
            let x = rng.gen();
            inputs_slice[i] = x;
            labels_slice[i] = x.sin();
        }

        (inputs, labels)
    }

    // Computes the mean squared error between 2 tensors.
    fn mse(backend: &Backend<Native>, in1: &SharedTensor<f32>, in2: &SharedTensor<f32>) -> f64 {
        let in1_slice = in1.read(backend.device()).unwrap().as_slice::<f32>();
        let in2_slice = in2.read(backend.device()).unwrap().as_slice::<f32>();
        assert_eq!(in1_slice.len(), in2_slice.len());

        in1_slice
            .iter()
            .zip(in2_slice.iter())
            .fold(0.0, |acc, (v1, v2)| acc + ((v1 - v2) * (v1 - v2)) as f64)
            .sqrt()
    }

    // Tests that the trainer can achieve convergence with the selected optimizer.
    fn test_convergence_with_optimizer(optimizer_cfg: OptimizerConfig) {
        let backend = native_backend();

        // Create network.
        let mut net_cfg = SequentialConfig::new();
        net_cfg.add_layer("linear1", LinearConfig::new(50));
        net_cfg.add_layer("relu1", LayerConfig::Relu);
        net_cfg.add_layer("linear2", LinearConfig::new(50));
        net_cfg.add_layer("relu2", LayerConfig::Relu);
        net_cfg.add_layer("linear3", LinearConfig::new(1));
        let mut net = Network::from_config(&backend, net_cfg, &[vec![1]]).unwrap();

        // Create trainer.
        let train_cfg = TrainerConfig {
            optimizer: optimizer_cfg,
            batch_size: BATCH_SIZE,
            ..Default::default()
        };
        let mut trainer = Trainer::from_config(&backend, train_cfg, &net, &vec![1]);

        // Iterate until we achieve 10X improvement over the initial MSE or reach the iteration limit.
        let mut initial_mse = None;
        let mut converged = false;
        for i in 0..10000 {
            let (inputs, labels) = create_batch_for_sin(&backend);
            let output = trainer.train_minibatch(&backend, &mut net, &inputs, &labels);
            let mse = mse(&backend, &output, &labels);

            if i % 100 == 0 {
                println!("MSE: {}", mse);
            }

            if i == 0 {
                initial_mse = Some(mse);
            }

            // Success condition (10X improvement).
            if let Some(m) = initial_mse {
                if mse < m * 0.1 {
                    converged = true;
                    break;
                }
            }
        }

        assert!(converged);
    }

    #[test]
    fn train_with_sgd_momentum() {
        test_convergence_with_optimizer(SgdWithMomentumConfig::default().into());
    }

    #[test]
    fn train_with_adam() {
        test_convergence_with_optimizer(AdamConfig::default().into());
    }
}
