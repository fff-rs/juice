extern crate coaster;
extern crate juice;

#[cfg(test)]
mod cartpole {
    /// This test verifies Q-learning in a cartpole environment:
    ///         O
    ///         │
    ///         │
    ///      ┌──┴──┐
    ///      │     ├───► F
    ///      └─────┘
    ///
    /// "Cartpole" is an inverted pendulum on a plaftorm that can move in one dimension.
    /// Agent must apply force to move the platform either to the left or to the right
    /// in order to balance the pole in an upright position. Scenario ends when the pole
    /// angle crosses a certain threshold. Agent is rewarded with R=1 for each cycle
    /// of the scenario (so the longer the agent is able to keep pole from falling, the bigger
    /// overall reward it gets).
    ///
    /// State "s" consists of [cart_pos, cart_vel, pole_angle, pole_angle_vel] variables.
    /// Possible actions "a" are [left, right].
    /// Q function Q(s, a) is approximated by a neural network with trainable weights θ: Q(s, a, θ).
    /// Network takes the 4 state variables as the input and outputs the the expected return value
    /// for each action:
    ///
    ///   [s1 s2 s3 s4] -> net -> [Q(s, left) Q(s, right)]
    ///
    /// During training, on each step agent observes the state "s", chooses the best action "a"
    /// using an ε-greedy policy based on currently learned Q-function (that is, it takes a random
    /// action with ε probability and an action "a" that maximizes Q(s, a) with probability 1-ε),
    /// gets the reward "R" and observes the next step "s'".
    /// The tuple [s, a, R, s'] is saved into experience replay buffer.
    ///
    /// For training, a batch is taken from the replay buffer. For each replay tuple [s, a, R, s'],
    /// "s" becomes the network input and the label is computed as
    ///
    ///   l = R + γ max Q*(s', a)
    ///              a
    ///
    /// where Q* is an earlier snapshot of Q (a snapshot is used instead of Q itself for stability).
    /// Note that this gives the label value only for one of the actions; the full label is set to
    ///
    ///   [l NaN]
    ///
    /// (assuming "a" was the first action). MSE loss function is assumed to ignore NaN values for
    /// loss and backpropagation gradient computations.
    use rand::{thread_rng, Rng};
    use std::collections::VecDeque;
    use std::rc::Rc;

    use coaster::frameworks::native::get_native_backend;
    use coaster::prelude::*;
    use juice::net::*;
    use juice::train::*;
    use juice::util::{write_batch_sample, LayerOps};

    const STATE_SIZE: usize = 4;
    const ACTION_COUNT: usize = 2;
    const BATCH_SIZE: usize = 32;
    const REPLAY_BUFFER_SIZE: usize = 1024;
    const CART_MASS: f64 = 1.0;
    const POLE_MASS: f64 = 1.0;
    const POLE_LENGTH: f64 = 1.0;
    const EARTH_G: f64 = 9.8;
    const ENV_STEP: f64 = 0.1;
    const DISCOUNT: f32 = 0.9;

    #[derive(Clone, Copy, Debug)]
    enum Action {
        Left,
        Right,
    }

    #[derive(Default)]
    struct Environment {
        // Position, velocity and acceleration of the cart.
        cart_pos: f64,
        cart_vel: f64,
        cart_acc: f64,

        // Angle, angular velocity and acceleration of the pole (angle = 0 means upright).
        pole_angle: f64,
        pole_angle_vel: f64,
        pole_angle_acc: f64,
    }

    struct ReplayEntry {
        state: [f32; STATE_SIZE],
        action: Action,
        reward: f32,
        // None if the resulting state is a final one.
        next_state: Option<[f32; STATE_SIZE]>,
    }

    impl Environment {
        fn new() -> Environment {
            Environment {
                // Start with a slight offset so that the pole starts falling.
                pole_angle: 0.001,
                ..Default::default()
            }
        }

        // Execute an actor action, transitioning into a new env state.
        // Action can be None for the purpose of testing the Environment logic itself
        // (agent always makes an action).
        fn step(&mut self, action: Option<Action>) {
            // Shorthands and local vars.
            let m_p = POLE_MASS;
            let m_c = CART_MASS;
            let l = POLE_LENGTH;
            let m = m_p + m_c;
            let f = match action {
                None => 0.0,
                Some(Action::Left) => -1.0,
                Some(Action::Right) => 1.0,
            };
            let th = self.pole_angle;
            let th_dot = self.pole_angle_vel;
            let th_ddot = self.pole_angle_acc;
            let sin_th = th.sin();
            let cos_th = th.cos();

            self.cart_acc = (f + m_p * l * (th_dot.powi(2) * sin_th - th_ddot * cos_th)) / m;
            self.cart_vel += self.cart_acc * ENV_STEP;
            self.cart_pos += self.cart_vel * ENV_STEP;

            self.pole_angle_acc = (EARTH_G * sin_th
                + cos_th * (-f - m_p * l * th_dot.powi(2) * sin_th) / m)
                / (l * (4.0 / 3.0 - m_p * cos_th.powi(2) / m));
            self.pole_angle_vel += self.pole_angle_acc * ENV_STEP;
            self.pole_angle += self.pole_angle_vel * ENV_STEP;
        }

        // Returns true if the pole has reached some critical angle.
        fn is_final(&self) -> bool {
            self.pole_angle.abs() > std::f64::consts::PI * 0.25
        }

        fn observe(&self) -> [f32; STATE_SIZE] {
            [
                self.cart_pos as f32,
                self.cart_vel as f32,
                self.pole_angle as f32,
                self.pole_angle_vel as f32,
            ]
        }
    }

    // Returns a completely random action.
    fn random_action() -> Action {
        if thread_rng().gen::<f64>() < 0.5 {
            Action::Left
        } else {
            Action::Right
        }
    }

    fn epsion_greedy_action<B: IBackend + LayerOps<f32> + 'static>(
        backend: &B,
        net: &Network<B>,
        state: &[f32; STATE_SIZE],
        epsilon: f64,
    ) -> Action {
        if thread_rng().gen::<f64>() < epsilon {
            random_action()
        } else {
            let action_values = get_action_values(backend, net, state);
            if action_values[0] > action_values[1] {
                Action::Left
            } else {
                Action::Right
            }
        }
    }

    // Returns the predicted action values for a given state.
    fn get_action_values<B: IBackend + LayerOps<f32> + 'static>(
        backend: &B,
        net: &Network<B>,
        state: &[f32; STATE_SIZE],
    ) -> [f32; ACTION_COUNT] {
        let mut input = SharedTensor::new(&[1, STATE_SIZE]);
        write_batch_sample(&mut input, state, 0);
        let output = net.transform(backend, &input);

        let mut result = [0.0; ACTION_COUNT];
        let native_backend = get_native_backend();
        result.clone_from_slice(output.read(native_backend.device()).unwrap().as_slice());
        result
    }

    fn create_batch<B: IBackend + LayerOps<f32> + 'static>(
        backend: &B,
        buffer: &VecDeque<ReplayEntry>,
        target_net: &Network<B>,
    ) -> (SharedTensor<f32>, SharedTensor<f32>) {
        let mut inputs = SharedTensor::new(&vec![BATCH_SIZE, STATE_SIZE]);
        let mut labels = SharedTensor::new(&vec![BATCH_SIZE, ACTION_COUNT]);

        for i in 0..BATCH_SIZE {
            let j = thread_rng().gen_range(0..buffer.len());
            let (buffer_action, other_action) = match buffer[j].action {
                Action::Left => (0, 1),
                Action::Right => (1, 0),
            };

            let mut action_values = [std::f32::NAN; ACTION_COUNT];

            // For the (s, a, s', r) tuple in the buffer, we can compute more precise target as
            //   y = r + γ•max Q*(s', _), if s' is not terminal, or
            //   y = r,                   if s' is termninal.
            action_values[buffer_action] = buffer[j].reward
                + match buffer[j].next_state {
                    None => 0.0,
                    Some(s) => {
                        DISCOUNT
                            * get_action_values(backend, target_net, &s)
                                .iter()
                                .fold(std::f32::NEG_INFINITY, |a, &b| a.max(b))
                    }
                };

            // For the other action a2, we just use the target_net:
            //   y = Q*(s, a2).
            // action_values[other_action] =
            //     get_action_values(backend, target_net, &buffer[j].state)[other_action];

            write_batch_sample(&mut inputs, &buffer[j].state, i);
            write_batch_sample(&mut labels, &action_values, i);
        }

        (inputs, labels)
    }

    // Runs 3 scenarios with a greedy policy using the provided learned Q-function.
    // Returns the average number of steps the agent was able to keep the cartpole from
    // falling (capped at 100 steps).
    fn eval<B: IBackend + LayerOps<f32> + 'static>(backend: &B, net: &Network<B>) -> f32 {
        let mut sum = 0.0;
        for _ in 0..3 {
            let mut env = Environment::new();
            for i in 0..100 {
                let action = epsion_greedy_action(backend, net, &env.observe(), 0.0);
                env.step(Some(action));
                sum += 1.0;
                if env.is_final() {
                    break;
                }
            }
        }
        sum / 3.0
    }

    // A test on the environment simulator.
    // When no forces are present, pole angle should be oscillating around PI.
    #[test]
    fn environment_is_sane_without_force() {
        let mut env = Environment::new();
        let mut avg_angle = 0.0;
        for _ in 0..10000 {
            env.step(None);
            avg_angle += env.pole_angle;
        }
        avg_angle /= 10000.0;
        assert!(
            (avg_angle - std::f64::consts::PI).abs() < 0.01,
            "Avg. angle: {}",
            avg_angle
        );
    }

    // A test on the environment simulator.
    // When force is applied to the left, cart should be moving to the left.
    #[test]
    fn environment_is_sane_with_left_force() {
        let mut env = Environment::new();
        for _ in 0..10 {
            env.step(Some(Action::Left));
        }
        assert!(env.cart_pos < 0.0, "Cart pos: {}", env.cart_pos);
        assert!(env.cart_vel < 0.0, "Cart vel: {}", env.cart_vel);
        assert!(env.cart_acc < 0.0, "Cart acc: {}", env.cart_acc);
    }

    // A test on the environment simulator.
    // When force is applied to the right, cart should be moving to the right.
    #[test]
    fn environment_is_sane_with_right_force() {
        let mut env = Environment::new();
        for _ in 0..10 {
            env.step(Some(Action::Right));
        }
        assert!(env.cart_pos > 0.0, "Cart pos: {}", env.cart_pos);
        assert!(env.cart_vel > 0.0, "Cart vel: {}", env.cart_vel);
        assert!(env.cart_acc > 0.0, "Cart acc: {}", env.cart_acc);
    }

    #[test]
    fn learns_cartpole_control() {
        env_logger::init();

        let backend = get_native_backend();

        // Create the network representing the Q-function Q(s, a).
        let net_conf = LayerConfig::Sequential(
            SequentialConfig::new()
                .with_layer("linear1", LayerConfig::Linear(LinearConfig::new(50)))
                .with_layer("relu1", LayerConfig::Relu)
                .with_layer("linear2", LayerConfig::Linear(LinearConfig::new(50)))
                .with_layer("relu2", LayerConfig::Relu)
                .with_layer(
                    "linear3",
                    LayerConfig::Linear(LinearConfig::new(ACTION_COUNT)),
                ),
        );
        let mut net = Network::from_config(net_conf, &[vec![STATE_SIZE]]);

        // Create the trainer.
        let trainer_conf = TrainerConfig {
            batch_size: BATCH_SIZE,
            objective: LayerConfig::MeanSquaredError,
            optimizer: OptimizerConfig::SgdWithMomentum(Default::default()),
            ..Default::default()
        };
        let mut trainer = Trainer::from_config(&backend, &trainer_conf, &net, &vec![ACTION_COUNT]);

        let mut replay_buffer = VecDeque::new();
        let mut env = Environment::new();

        // Network used to compute full returns from a certain state.
        // This is a periodic snapshot from the main net (main network isn't used due to
        // stability issues).
        let mut target_net = net.clone();
        let mut epsilon = 1.0;

        for i in 0..1000000 {
            // Do a step.
            let state = env.observe();
            let action = epsion_greedy_action(&backend, &net, &state, epsilon);
            env.step(Some(action));
            let (reward, next_state) = match env.is_final() {
                false => (1.0, Some(env.observe())),
                true => (0.0, None),
            };

            // Store the result in the replay buffer.
            let replay_entry = ReplayEntry {
                state: state,
                action: action,
                reward: reward,
                next_state: next_state,
            };
            replay_buffer.push_front(replay_entry);
            replay_buffer.truncate(REPLAY_BUFFER_SIZE);

            // Restart the environment if reached a final state.
            if env.is_final() {
                env = Environment::new();
            }

            if replay_buffer.len() >= BATCH_SIZE {
                let (inputs, labels) = create_batch(&backend, &replay_buffer, &target_net);
                trainer.train_minibatch(&backend, &mut net, &inputs, &labels);
            }

            // Evaluate performance and snapshot the bootstrapping net every 100 steps.
            if i % 100 == 0 {
                let score = eval(&backend, &net);
                println!("Epoch: {}; score: {}; ε: {}", i / 100, score, epsilon);
                target_net = net.clone();
                epsilon = (epsilon * 0.995).max(0.01);

                // Stop when we reach 95 score.
                if score >= 95.0 {
                    return;
                }
            }
        }

        assert!(false, "Failed to reach score 95");
    }
}
