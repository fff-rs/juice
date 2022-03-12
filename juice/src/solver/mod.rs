//! Provides the generics and interfaces for the specific Solvers.
//!
//! See [Solvers][solvers]
//! [solvers]: ../solvers/index.html

pub mod confusion_matrix;
pub mod regression_evaluator;

use std::cell::RefCell;

use std::rc::Rc;

pub use self::confusion_matrix::ConfusionMatrix;
pub use self::regression_evaluator::{RegressionEvaluator, RegressionLoss};
use crate::co::prelude::*;
use crate::net::*;
use crate::solvers::*;

use crate::util::{ArcLock, LayerOps, SolverOps};

#[derive(Debug)]
/// Solver that optimizes a [Layer][1] with a given objective.
/// [1]: ../layer/index.html
pub struct Solver<B>
where
    B: IBackend + LayerOps<f32>,
{
    backend: Rc<B>,
    context: Context,

    net: Box<dyn Layer<B>>,
    objective: Box<dyn Layer<B>>,
    /// The implementation of the Solver
    pub worker: Box<dyn ISolver<B, B>>,

    config: SolverConfig,

    /// The current iteration / number of times weights have been updated
    iter: usize,
}

impl<B> Solver<B>
where
    B: IBackend + LayerOps<f32> + SolverOps<f32> + 'static,
{
    /// Create Solver from [SolverConfig][1]
    /// [1]: ./struct.SolverConfig.html
    ///
    /// This is the **preferred method** to create a Solver for training a neural network.
    pub fn from_config(
        backend: Rc<B>,
        config: &SolverConfig,
        input_shapes: &[TensorDesc],
        label_shape: &TensorDesc,
    ) -> Solver<B> {
        let context = Context::new(config.minibatch_size);
        let network = layer_from_config(
            Descriptor::top(
                "net",
                input_shapes
                    .iter()
                    .enumerate()
                    .map(|(i, shape)| Inout::new_with_path(shape.clone(), &format!("net_in_{}", i)))
                    .collect(),
            ),
            &config.network,
        );
        assert_eq!(network.descriptor().outputs().len(), 1); // Net must have only one output.

        let objective = layer_from_config(
            Descriptor::top(
                "loss",
                vec![
                    network.descriptor().output(0).clone(),
                    Inout::new_with_path(label_shape.clone(), "labels"),
                ],
            ),
            &config.objective,
        );

        let weight_shapes: Vec<TensorDesc> = network
            .descriptor()
            .params()
            .iter()
            .map(|w| w.borrow().data.desc().clone())
            .collect();
        let mut worker = config.solver.with_config(backend.clone(), &config);
        worker.init(&weight_shapes);

        // Loss layer cannot have params (no one will train them!).
        assert!(objective.descriptor().params().is_empty());

        Solver {
            backend: backend,
            context: context,
            worker: worker,
            net: network,
            objective: objective,
            iter: 0,
            config: config.clone(),
        }
    }
}

impl<B> Solver<B>
where
    B: IBackend + LayerOps<f32> + SolverOps<f32> + 'static,
{
    fn init(&mut self, backend: Rc<B>, input_shapes: &[TensorDesc]) {
        info!("Initializing solver from configuration");

        let config = self.config.clone();
        self.init_net(backend, &config, input_shapes);
    }

    /// Initialize the training net
    fn init_net(&mut self, backend: Rc<B>, param: &SolverConfig, input_shapes: &[TensorDesc]) {
        unimplemented!();
        //self.net = layer_from_config(&*backend, &param.network, input_shapes);
    }

    /// Train the network with one minibatch
    pub fn train_minibatch(
        &mut self,
        mb_data: ArcLock<SharedTensor<f32>>,
        mb_target: ArcLock<SharedTensor<f32>>,
    ) -> SharedTensor<f32> {
        // Copy intput data into the network context.
        let data = self.context.acquire_data(self.net.descriptor().input(0));
        self.backend
            .copy(&mb_data.read().unwrap(), &mut data.borrow_mut());
        let labels = self
            .context
            .acquire_data(self.objective.descriptor().input(1));
        self.backend
            .copy(&mb_target.read().unwrap(), &mut labels.borrow_mut());

        // Compute network output and the loss.
        self.net.compute_output(&*self.backend, &mut self.context);
        self.objective
            .compute_output(&*self.backend, &mut self.context);

        // Compute params gradients by doing a backpropagation on the network.
        self.objective
            .compute_gradients(&*self.backend, &mut self.context);
        self.net
            .compute_gradients(&*self.backend, &mut self.context);

        // Let the solver worker adjust the params gradients before applying them to params.
        let params: Vec<LearnableParamsLink> =
            self.net.descriptor().params().iter().cloned().collect();
        let params_gradients: Vec<(Rc<RefCell<SharedTensor<f32>>>, f32)> = params
            .iter()
            .map(|p| {
                (
                    self.context.get_params_gradient(p),
                    p.borrow().learning_rate,
                )
            })
            .collect();
        self.worker
            .compute_update(&self.config, &params_gradients, self.iter);

        // Finally apply the weight change.
        let shared_a = crate::util::native_scalar(-1f32);
        for i in 0..self.net.descriptor().params().len() {
            let gradient = &params_gradients[i].0.borrow();
            let params = &mut self.net.descriptor_mut().param(i).borrow_mut().data;
            self.backend.axpy(&shared_a, gradient, params).unwrap();
        }

        self.iter += 1;

        let out_buffer = self.context.get_data(self.net.descriptor().output(0));
        let mut network_out = SharedTensor::<f32>::new(out_buffer.borrow().desc());
        self.backend.copy(&out_buffer.borrow(), &mut network_out);
        network_out
    }

    // /// Returns the network trained by the solver.
    // ///
    // /// This is the recommended method to get a usable trained network.
    // pub fn network(&self) -> &Layer<B> {
    //     &self.net
    // }

    // /// Returns the network trained by the solver.
    // ///
    // /// This is the recommended method to get a trained network,
    // /// if you want to alter the network. Keep in mind that altering the network
    // /// might render the solver unusable and continuing training the network with it will yield
    // /// unexpected results.
    // pub fn mut_network(&mut self) -> &mut Layer<B> {
    //     &mut self.net
    // }
}

/// Implementation of a specific Solver.
///
/// See [Solvers][1]
/// [1]: ../solvers/index.html
pub trait ISolver<SolverB, B>
where
    B: IBackend + LayerOps<f32>,
    SolverB: IBackend + SolverOps<f32>,
{
    /// Initialize the solver, setting up any network related data.
    fn init(&mut self, weight_shapes: &[TensorDesc]) {}

    /// Update the weights of the net with part of the gradient.
    ///
    /// The [second phase of backpropagation learning][1].
    /// Calculates the gradient update that should be applied to the network,
    /// and then applies that gradient to the network, changing its weights.
    ///
    /// [1]: https://en.wikipedia.org/wiki/Backpropagation#Phase_2:_Weight_update
    ///
    /// Used by [step][2] to optimize the network.
    ///
    /// [2]: ./struct.Solver.html#method.step
    fn compute_update(
        &mut self,
        param: &SolverConfig,
        weight_gradients: &[(Rc<RefCell<SharedTensor<f32>>>, f32)],
        iter: usize,
    );

    /// Returns the backend used by the solver.
    fn backend(&self) -> &SolverB;
}

impl<SolverB, B: IBackend + LayerOps<f32>> ::std::fmt::Debug for dyn ISolver<SolverB, B> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "({})", "ILayer")
    }
}

#[derive(Debug, Clone)]
/// Configuration for a Solver
pub struct SolverConfig {
    /// Name of the solver.
    pub name: String,
    /// The [LayerConfig][1] that is used to initialize the network.
    /// [1]: ../layer/struct.LayerConfig.html
    pub network: LayerConfig,
    /// The [LayerConfig][1] that is used to initialize the objective.
    /// [1]: ../layer/struct.LayerConfig.html
    pub objective: LayerConfig,
    /// The [Solver implementation][1] to be used.
    /// [1]: ../solvers/index.html
    pub solver: SolverKind,
    /// Accumulate gradients over `minibatch_size` instances.
    ///
    /// Default: 1
    pub minibatch_size: usize,
    /// The learning rate policy to be used.
    ///
    /// Default: Fixed
    pub lr_policy: LRPolicy,
    /// The base learning rate.
    ///
    /// Default: 0.01
    pub base_lr: f32,
    /// gamma as used in the calculation of most learning rate policies.
    ///
    /// Default: 0.1
    pub gamma: f32,
    /// The stepsize used in Step and Sigmoid learning policies.
    ///
    /// Default: 10
    pub stepsize: usize,
    /// The threshold for clipping gradients.
    ///
    /// Gradient values will be scaled to their [L2 norm][1] of length `clip_gradients`
    /// if their L2 norm is larger than `clip_gradients`.
    /// If set to `None` gradients will not be clipped.
    ///
    /// [1]: https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm
    ///
    /// Default: None
    pub clip_gradients: Option<f32>,
    /// The global [weight decay][1] multiplier for [regularization][2].
    /// [1]: http://www.alglib.net/dataanalysis/improvinggeneralization.php#header3
    /// [2]: https://cs231n.github.io/neural-networks-2/#reg
    ///
    /// Regularization can prevent [overfitting][3].
    ///
    /// If set to `None` no regularization will be performed.
    ///
    /// [3]: https://cs231n.github.io/neural-networks-2/#reg
    pub weight_decay: Option<f32>,
    /// The method of [regularization][1] to use.
    /// [1]: https://cs231n.github.io/neural-networks-2/#reg
    ///
    /// There are different methods for regularization.
    /// The two most common ones are [L1 regularization][1] and [L2 regularization][1].
    ///
    /// See [RegularizationMethod][2] for all implemented methods.
    ///
    /// [2]: ./enum.RegularizationMethod.html
    ///
    /// Currently only L2 regularization is implemented.
    /// See [Issue #23](https://github.com/spearow/juice/issues/23).
    pub regularization_method: Option<RegularizationMethod>,
    /// The [momentum][1] multiplier for [SGD solvers][2].
    /// [1]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum
    /// [2]: ../solvers/sgd/index.html
    ///
    /// For more information see [SGD with momentum][3]
    /// [3]: ../solvers/sgd/momentum/index.html
    ///
    /// The value should always be between 0 and 1 and dictates how much of the previous
    /// gradient update will be added to the current one.
    ///
    /// Default: 0
    pub momentum: f32,
}

impl Default for SolverConfig {
    fn default() -> SolverConfig {
        SolverConfig {
            name: "".to_owned(),
            network: LayerConfig::default(),
            objective: LayerConfig::default(),
            solver: SolverKind::SGD(SGDKind::Momentum),

            minibatch_size: 1,

            lr_policy: LRPolicy::Fixed,
            base_lr: 0.01f32,
            gamma: 0.1f32,
            stepsize: 10,

            clip_gradients: None,

            weight_decay: None,
            regularization_method: None,

            momentum: 0f32,
        }
    }
}

impl SolverConfig {
    /// Return the learning rate for a supplied iteration.
    ///
    /// The way the learning rate is calculated depends on the configured [LRPolicy][1].
    ///
    /// [1]: ./enum.LRPolicy.html
    ///
    /// Used by the [Solver][2] to calculate the learning rate for the current iteration.
    /// The calculated learning rate has a different effect on training dependent on what
    /// [type of Solver][3] you are using.
    ///
    /// [2]: ./struct.Solver.html
    /// [3]: ../solvers/index.html
    pub fn get_learning_rate(&self, iter: usize) -> f32 {
        match self.lr_policy() {
            LRPolicy::Fixed => self.base_lr(),
            LRPolicy::Step => {
                let current_step = self.step(iter);
                self.base_lr() * self.gamma().powf(current_step as f32)
            }
            // LRPolicy::Multistep => {
            //     // TODO: the current step can be calculated on-demand
            //     //   if (this->current_step_ < this->param_.stepvalue_size() &&
            //     //         this->iter_ >= this->param_.stepvalue(this->current_step_)) {
            //     //     this->current_step_++;
            //     //     LOG(INFO) << "MultiStep Status: Iteration " <<
            //     //     this->iter_ << ", step = " << this->current_step_;
            //     //   }
            //     //   rate = this->param_.base_lr() *
            //     //       pow(this->param_.gamma(), this->current_step_);
            //     unimplemented!();
            // }
            LRPolicy::Exp => self.base_lr() * self.gamma().powf(iter as f32),
            // LRPolicy::Inv => {
            //     //   rate = this->param_.base_lr() *
            //     //       pow(Dtype(1) + this->param_.gamma() * this->iter_,
            //     //           - this->param_.power());
            //     unimplemented!();
            // }
            // LRPolicy::Poly => {
            //     //   rate = this->param_.base_lr() * pow(Dtype(1.) -
            //     //       (Dtype(this->iter_) / Dtype(this->param_.max_iter())),
            //     //       this->param_.power());
            //     unimplemented!();
            // }
            // LRPolicy::Sigmoid => {
            //     //   rate = this->param_.base_lr() * (Dtype(1.) /
            //     //       (Dtype(1.) + exp(-this->param_.gamma() * (Dtype(this->iter_) -
            //     //         Dtype(this->param_.stepsize())))));
            //     unimplemented!();
            // }
        }
    }

    /// Return current step at iteration `iter`.
    ///
    /// Small helper for learning rate calculation.
    fn step(&self, iter: usize) -> usize {
        iter / self.stepsize()
    }

    /// Return learning rate policy.
    fn lr_policy(&self) -> LRPolicy {
        self.lr_policy
    }

    /// Return the base learning rate.
    fn base_lr(&self) -> f32 {
        self.base_lr
    }

    /// Return the gamma for learning rate calculations.
    fn gamma(&self) -> f32 {
        self.gamma
    }

    /// Return the stepsize for learning rate calculations.
    fn stepsize(&self) -> usize {
        self.stepsize
    }
}

#[derive(Debug, Copy, Clone)]
/// All available types of solvers.
pub enum SolverKind {
    /// Stochastic Gradient Descent.
    /// See [SGDKind][1] for all available SGD solvers.
    /// [1]: ./enum.SGDKind.html
    SGD(SGDKind),
}

impl SolverKind {
    /// Create a Solver of the specified kind with the supplied SolverConfig.
    pub fn with_config<
        B: IBackend + SolverOps<f32> + 'static,
        NetB: IBackend + LayerOps<f32> + 'static,
    >(
        &self,
        backend: Rc<B>,
        config: &SolverConfig,
    ) -> Box<dyn ISolver<B, NetB>> {
        match *self {
            SolverKind::SGD(sgd) => sgd.with_config(backend, config),
        }
    }
}

#[derive(Debug, Copy, Clone)]
/// All available types of Stochastic Gradient Descent solvers.
pub enum SGDKind {
    /// Stochastic Gradient Descent with Momentum. See [implementation][1]
    /// [1] ../solvers/
    Momentum,
}

impl SGDKind {
    /// Create a Solver of the specified kind with the supplied SolverConfig.
    pub fn with_config<
        B: IBackend + SolverOps<f32> + 'static,
        NetB: IBackend + LayerOps<f32> + 'static,
    >(
        &self,
        backend: Rc<B>,
        config: &SolverConfig,
    ) -> Box<dyn ISolver<B, NetB>> {
        match *self {
            SGDKind::Momentum => Box::new(Momentum::<B>::new(backend)),
        }
    }
}

#[derive(Debug, Copy, Clone)]
/// Learning Rate Policy for a [Solver][1]
/// [1]: ./struct.Solver.html
///
/// The variables mentioned below are defined in the [SolverConfig][2] apart from
/// iter, which is the current iteration of the solver, that is supplied as a parameter
/// for the learning rate calculation.
///
/// [2]: ./struct.SolverConfig.html
pub enum LRPolicy {
    /// always return base_lr
    Fixed,
    /// learning rate decays every `step` iterations.
    /// return base_lr * gamma ^ (floor(iter / step))
    Step,
    // /// similar to step but it allows non uniform steps defined by
    // /// stepvalue
    // Multistep,
    /// return base_lr * gamma ^ iter
    Exp,
    // /// return base_lr * (1 + gamma * iter) ^ (- power)
    // Inv,
    // /// the effective learning rate follows a polynomial decay, to be
    // /// zero by the max_iter.
    // /// return base_lr (1 - iter/max_iter) ^ (power)
    // Poly,
    // /// the effective learning rate follows a sigmod decay
    // /// return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
    // Sigmoid,
}

#[derive(Debug, Copy, Clone)]
/// [Regularization][1] method for a [Solver][2].
/// [1]: https://cs231n.github.io/neural-networks-2/#reg
/// [2]: ./struct.Solver.html
pub enum RegularizationMethod {
    /// L2 regularization
    L2,
}
