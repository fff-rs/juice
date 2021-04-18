#![allow(unused_must_use)]

use coaster as co;
use coaster_nn as conn;

use fs_err::File;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

use csv::Reader;
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[cfg(all(feature = "cuda"))]
use co::frameworks::cuda::get_cuda_backend;
use co::prelude::*;
use conn::{DirectionMode, RnnInputMode, RnnNetworkMode};
use juice::layer::*;
use juice::layers::*;
use juice::solver::*;
use juice::util::*;

const MAIN_USAGE: &str = "
Demonstrate RNN caps of juice with the cuda backend.

Usage:
    mackey-glass-example train [--batch-size=<batch>] [--learning-rate=<lr>] [--momentum=<f>] <networkfile>
    mackey-glass-example test <networkfile>

Options:
    -b, --batch-size=<batch>    Network Batch Size.
    -l, --learning-rate=<lr>    Learning Rate.
    -m, --momentum=<f>         Momentum.
    -h, --help                 Show this screen.
";

#[allow(non_snake_case)]
#[derive(Deserialize, Debug, Default)]
struct Args {
    cmd_train: bool,
    cmd_test: bool,
    flag_batch_size: Option<usize>,
    flag_learning_rate: Option<f32>,
    flag_momentum: Option<f32>,
    /// Path to the stored network.
    arg_networkfile: PathBuf,
}

impl Args {
    pub(crate) fn data_mode(&self) -> DataMode {
        assert_ne!(self.cmd_train, self.cmd_test);
        if self.cmd_train {
            return DataMode::Train
        }
        if self.cmd_test {
            return DataMode::Test
        }
        unreachable!("nope");
    }
}

const fn default_learning_rate() -> f32 {
    0.10_f32
}

const fn default_momentum() -> f32 {
    0.00
}

const fn default_batch_size() -> usize {
    10
}

impl std::cmp::PartialEq for Args {
    fn eq(&self, other: &Self) -> bool {
        match (self.flag_learning_rate, other.flag_learning_rate) {
            (Some(lhs), Some(rhs)) if (rhs - lhs).abs() < 1e6 => {}
            (None, None) => {},
            _ => return false,
        }
        match (self.flag_momentum, other.flag_momentum) {
            (Some(lhs), Some(rhs)) if (rhs - lhs).abs() < 1e6 => {}
            (None, None) => {},
            _ => return false,
        }
        self.cmd_test == other.cmd_test &&
        self.cmd_train == other.cmd_train &&
        self.arg_networkfile == other.arg_networkfile &&
        self.flag_batch_size == other.flag_batch_size
    }
}

impl std::cmp::Eq for Args {}

enum DataMode {
    Train,
    Test,
}

impl DataMode {
    fn as_path(&self) -> &'static str {
        match self {
            DataMode::Train => "assets/norm_mackeyglass_train.csv",
            DataMode::Test => "assets/norm_mackeyglass_test.csv",
        }
    }
}


mod work {

use super::*;

const TRAIN_ROWS: usize = 35192;
const TEST_ROWS: usize = 8798;
const DATA_COLUMNS: usize = 10;

// Provide an Iterator over the input data
fn data_generator(data: DataMode) -> impl Iterator<Item = (f32, Vec<f32>)> {
    let rdr = Reader::from_reader(File::open(data.as_path()).unwrap());
    rdr.into_deserialize()
        .map(move |row : Result<Vec<f32>, _>| match row {
        Ok(row_vec) => {
            let label = row_vec[0];
            let columns = row_vec[1..=DATA_COLUMNS].to_vec();
            (label, columns)
        }
        _ => {
            log::error!("file seems to be empty");
            panic!();
        }
    })
}

fn create_network(batch_size: usize, columns: usize) -> SequentialConfig {
    // Create a simple Network
    // * LSTM Layer
    // * Single Neuron
    // * Sigmoid Activation Function

    let mut net_cfg = SequentialConfig::default();
    // The input is a 3D Tensor with Batch Size, Rows, Columns. Columns are already ordered
    // and it is expected that the RNN move across them using this order.
    net_cfg.add_input("data_input", &[batch_size, 1_usize, columns]);
    net_cfg.force_backward = true;

    // Reshape the input into NCHW Format
    net_cfg.add_layer(LayerConfig::new(
        "reshape",
        LayerType::Reshape(ReshapeConfig::of_shape(&[batch_size, DATA_COLUMNS, 1, 1])),
    ));

    net_cfg.add_layer(LayerConfig::new(
        // Layer name is only used internally - can be changed to anything
        "LSTMInitial",
        RnnConfig {
            hidden_size: 5,
            num_layers: 2,
            dropout_seed: 123,
            dropout_probability: 0.5,
            rnn_type: RnnNetworkMode::LSTM,
            input_mode: RnnInputMode::LinearInput,
            direction_mode: DirectionMode::UniDirectional,
        },
    ));
    net_cfg.add_layer(LayerConfig::new("linear1", LinearConfig { output_size: 1 }));
    net_cfg.add_layer(LayerConfig::new("sigmoid", LayerType::Sigmoid));
    net_cfg
}

fn add_solver(
    backend: Rc<Backend<Cuda>>,
    net_cfg: SequentialConfig,
    batch_size: usize,
    learning_rate: f32,
    momentum: f32,
) -> Solver<Backend<Cuda>, Backend<Cuda>> {
    // Define an Objective Function
    let mut regressor_cfg = SequentialConfig::default();

    // Bit confusing, but the output is seen as the same as the input?
    regressor_cfg.add_input("data_output", &[batch_size, 1]);
    regressor_cfg.add_input("label", &[batch_size, 1]);

    // Add a Layer expressing Mean Squared Error (MSE) Loss. This will be used with the solver to
    // train the model.
    let mse_layer_cfg = LayerConfig::new("mse", LayerType::MeanSquaredError);
    regressor_cfg.add_layer(mse_layer_cfg);

    // Setup an Optimiser
    let mut solver_cfg = SolverConfig {
        minibatch_size: batch_size,
        base_lr: learning_rate,
        momentum,
        ..SolverConfig::default()
    };

    solver_cfg.network = LayerConfig::new("network", net_cfg);
    solver_cfg.objective = LayerConfig::new("regressor", regressor_cfg);
    Solver::from_config(backend.clone(), backend, &solver_cfg)
}

/// Train, and optionally, save the resulting network state/weights
pub(crate) fn train(
    backend: Rc<Backend<Cuda>>,
    batch_size: usize,
    learning_rate: f32,
    momentum: f32,
    file: &PathBuf,
) {
    // Initialise a Sequential Layer
    let net_cfg = create_network(batch_size, DATA_COLUMNS);
    let mut solver = add_solver(backend, net_cfg,  batch_size, learning_rate, momentum);

    // Define Input & Labels
    let input = SharedTensor::<f32>::new(&[batch_size, 1, DATA_COLUMNS]);
    let input_lock = Arc::new(RwLock::new(input));

    let label = SharedTensor::<f32>::new(&[batch_size, 1]);
    let label_lock = Arc::new(RwLock::new(label));

    // Define Evaluation Method - Using Mean Squared Error
    let mut regression_evaluator =
        ::juice::solver::RegressionEvaluator::new(Some("mse".to_owned()));
    // Indicate how many samples to average loss over
    regression_evaluator.set_capacity(Some(2000));

    let mut data_rows = data_generator(DataMode::Train);
    for _ in 0..(TRAIN_ROWS / batch_size) {
        let mut targets = Vec::new();
        for (batch_n, (label_val, input)) in data_rows.by_ref().take(batch_size).enumerate() {
            let mut input_tensor = input_lock.write().unwrap();
            let mut label_tensor = label_lock.write().unwrap();
            write_batch_sample(&mut input_tensor, &input, batch_n);
            write_batch_sample(&mut label_tensor, &[label_val], batch_n);
            targets.push(label_val);
        }
        // Train the network
        let inferred_out = solver.train_minibatch(input_lock.clone(), label_lock.clone());
        let mut inferred = inferred_out.write().unwrap();
        let predictions = regression_evaluator.get_predictions(&mut inferred);
        regression_evaluator.add_samples(&predictions, &targets);
        println!(
            "Mean Squared Error {}",
            &regression_evaluator.accuracy() as &dyn RegressionLoss
        );
    }

    solver.mut_network().save(file).unwrap();

}


/// Test a the validation subset of data items against the trained network state.
pub(crate) fn test(backend: Rc<Backend<Cuda>>, batch_size: usize, file: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Load in a pre-trained network
    let mut network: Layer<Backend<Cuda>> = Layer::<Backend<Cuda>>::load(backend, file)?;

    // Define Input & Labels
    let input = SharedTensor::<f32>::new(&[batch_size, 1, DATA_COLUMNS]);
    let input_lock = Arc::new(RwLock::new(input));

    let label = SharedTensor::<f32>::new(&[batch_size, 1]);
    let label_lock = Arc::new(RwLock::new(label));

    // Define Evaluation Method - Using Mean Squared Error
    let mut regression_evaluator =
        ::juice::solver::RegressionEvaluator::new(Some("mse".to_owned()));
    // Indicate how many samples to average loss over
    regression_evaluator.set_capacity(Some(2000));

    let mut data_rows = data_generator(DataMode::Test);

    for _ in 0..(TEST_ROWS / batch_size) {
        let mut targets = Vec::new();
        for (batch_n, (label_val, input)) in data_rows.by_ref().take(batch_size).enumerate() {
            let mut input_tensor = input_lock.write().unwrap();
            let mut label_tensor = label_lock.write().unwrap();
            write_batch_sample(&mut input_tensor, &input, batch_n);
            write_batch_sample(&mut label_tensor, &[label_val], batch_n);
            targets.push(label_val);
        }
        let results_vec = network.forward(&[input_lock.clone()]);
        let mut results = results_vec.get(0).unwrap().write().unwrap();
        let predictions = regression_evaluator.get_predictions(&mut results);
        regression_evaluator.add_samples(&predictions, &targets);
        println!(
            "Mean Squared Error {}",
            &regression_evaluator.accuracy() as &dyn RegressionLoss
        );
    }
    Ok(())
}

}

fn main() {
    env_logger::builder().filter_level(log::LevelFilter::Trace).init();
    // Parse Arguments
    let args: Args = docopt::Docopt::new(MAIN_USAGE)
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());

    #[cfg(all(feature = "cuda"))]
    {

        // Initialise a CUDA Backend, and the CUDNN and CUBLAS libraries.
        let backend = Rc::new(get_cuda_backend());

        match args.data_mode() {
            DataMode::Train =>
                work::train(
                    backend,
                    args.flag_batch_size.unwrap_or(default_batch_size()),
                    args.flag_learning_rate.unwrap_or(default_learning_rate()),
                    args.flag_momentum.unwrap_or(default_momentum()),
                    &args.arg_networkfile,
                ),
            DataMode::Test =>
                work::test(
                    backend,
                    args.flag_batch_size.unwrap_or(default_batch_size()),
                    &args.arg_networkfile).unwrap(),
        }
    }
    #[cfg(not(feature = "cuda"))]
    panic!("Juice currently only supports RNNs via CUDA & CUDNN. If you'd like to check progress \
                on native support, please look at the tracking issue https://github.com/spearow/juice/issues/41 \
                or the 2021/2022 road map https://github.com/spearow/juice/issues/30")
}


#[test]
fn docopt_works() {
    let check = |args: &[&'static str], expected: Args| {
        let docopt = docopt::Docopt::new(MAIN_USAGE).expect("Docopt spec if valid. qed");
        let args: Args = docopt.argv(args).deserialize().expect("Must deserialize. qed");

        assert_eq!(args, expected, "Expectations of {:?} stay unmet.", args);
    };

    check(&["mackey-glass-example", "train", "--learning-rate=0.4", "--batch-size=11", "--momentum=0.17", "ahoi.capnp"], Args {
        cmd_train: true,
        cmd_test: false,
        flag_batch_size: Some(11),
        flag_learning_rate: Some(0.4_f32),
        flag_momentum: Some(0.17_f32),
        arg_networkfile: PathBuf::from("ahoi.capnp"),
    });


    check(&["mackey-glass-example", "test", "ahoi.capnp"], Args {
        cmd_train: false,
        cmd_test: true,
        flag_batch_size: None,
        flag_learning_rate: None,
        flag_momentum: None,
        arg_networkfile: PathBuf::from("ahoi.capnp"),
    });
}
