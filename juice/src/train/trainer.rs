use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::co::prelude::*;
use crate::net::*;
use crate::train::{optimizer_from_config, Optimizer, OptimizerConfig, SgdWithMomentumConfig};
use crate::util::{format_tensor, SolverOps};

#[derive(Clone)]
pub struct TrainerConfig {
    pub batch_size: usize,
    pub objective: LayerConfig,
    pub optimizer: OptimizerConfig,
    pub learning_rate: f32,
}

/// Trains a network through minibatch backpropagation.
///
/// Trains a network doing backpropagation from a configured output. For multi-output
/// networks, several Trainers can be constructed (each with its own loss function)
/// to perform asynchronous training.
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
    pub fn from_config(
        backend: &B,
        config: &TrainerConfig,
        net: &Network<B>,
        label_shape: &TensorDesc,
    ) -> Self {
        // Create objective.
        let objective_descriptor = Descriptor::top(
            "loss",
            vec![
                net.top().descriptor().output(0).clone(),
                Inout::new_with_path(label_shape.clone(), "labels"),
            ],
        );
        let objective = layer_from_config(objective_descriptor, &config.objective);
        let optimizer = optimizer_from_config(&config.optimizer);

        Trainer {
            config: (*config).clone(),
            objective: objective,
            optimizer: optimizer,
            iter: 0,
        }
    }

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
        backend
            .copy(&inputs, &mut context_inputs.borrow_mut())
            .unwrap();
        // Copy labels.
        let context_labels = context.acquire_data(self.objective.descriptor().input(1));
        backend
            .copy(&labels, &mut context_labels.borrow_mut())
            .unwrap();

        // Compute network output and the loss.
        net.top().compute_output(backend, &mut context);
        self.objective.compute_output(backend, &mut context);

        // Compute params gradients by doing a backpropagation on the network.
        self.objective.compute_gradients(backend, &mut context);
        trace!(
            "Loss gradient:\n{}",
            format_tensor(
                &context
                    .get_data_gradient(self.objective.descriptor().input(0))
                    .borrow()
            )
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
        self.optimizer
            .adjust_weight_change(backend, &params_gradients);

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
