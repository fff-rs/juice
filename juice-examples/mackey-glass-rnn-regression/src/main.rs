extern crate coaster as co;
extern crate coaster_nn as conn;
extern crate env_logger;
extern crate juice;

use std::fs::File;
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

#[cfg(all(feature = "cuda"))]
use co::frameworks::cuda::get_cuda_backend;
use co::prelude::*;
use conn::{DirectionMode, RnnInputMode, RnnNetworkMode};
use csv::Reader;
use juice::layer::*;
use juice::layers::*;
use juice::solver::*;
use juice::util::*;
use serde::Deserialize;


const MAIN_USAGE: &str = "
Usage:  mackey-glass-example train [--file=<path>] [--batchSize=<batch>] [--learningRate=<lr>] [--momentum=<float>]
        mackey-glass-example test [--file=<path>]

Options:
    --file=<path>  Filepath for saving trained network.
    --batchSize=<batch>  Network Batch Size.
    --learningRate=<lr>  Learning Rate.
    --momentum=<float>  Momentum.
    -h, --help  Show this screen.
";

#[allow(non_snake_case)]
#[derive(Deserialize, Debug)]
struct Args {
    cmd_train: bool,
    cmd_test: bool,
    flag_file: Option<String>,
    #[allow(non_snake_case)]
    flag_batchSize: Option<usize>,
    #[allow(non_snake_case)]
    flag_learningRate: Option<f32>,
    flag_momentum: Option<f32>,
}

// Provide an Iterator over the input data
fn data_generator() -> impl Iterator<Item=(f32, Vec<f32>)> {
    let rdr = Reader::from_reader(File::open("assets/normalised_mackeyglass_lstm.csv").unwrap());
    let columns: usize = 10;

    rdr.into_deserialize().map(move |row| match row {
        Ok(value) => {
            let row_vec: Box<Vec<f32>> = Box::new(value);
            let label = row_vec[0];
            let columns = row_vec[1..=columns].to_vec();
            (label, columns)
        }
        _ => {
            println!("no value");
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

    net_cfg.add_layer(LayerConfig::new(
        // Layer name is only used internally - can be changed to anything
        "LSTMInitial",
        RnnConfig {
            hidden_size: 200,
            num_layers: 10,
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

#[cfg(all(feature = "cuda"))]
fn add_solver(
    net_cfg: SequentialConfig,
    backend: Rc<Backend<Cuda>>,
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

#[cfg(all(feature = "cuda"))]
#[allow(dead_code)]
fn train(
    batch_size: Option<usize>,
    learning_rate: Option<f32>,
    momentum: Option<f32>,
    file: Option<String>,
) {
    // Initialise a CUDA Backend, and the CUDNN and CUBLAS libraries.
    let backend = Rc::new(get_cuda_backend());

    let example_count: usize = 44000;
    let columns: usize = 10;

    let batch_size = batch_size.unwrap_or(200);
    let learning_rate = learning_rate.unwrap_or(0.01f32);
    let momentum = momentum.unwrap_or(0.00f32);

    // Initialise a Sequential Layer
    let net_cfg = create_network(batch_size, columns);
    let mut solver = add_solver(net_cfg, backend, batch_size, learning_rate, momentum);

    // Define Input & Labels
    let input = SharedTensor::<f32>::new(&[batch_size, 1, columns]);
    let input_lock = Arc::new(RwLock::new(input));

    let label = SharedTensor::<f32>::new(&[batch_size, 1]);
    let label_lock = Arc::new(RwLock::new(label));

    // Define Evaluation Method - Using Mean Squared Error
    let mut regression_evaluator =
        ::juice::solver::RegressionEvaluator::new(Some("mse".to_owned()));
    // Indicate how many samples to average loss over
    regression_evaluator.set_capacity(Some(2000));

    let mut data_rows = data_generator();
    for _ in 0..(example_count / batch_size) {
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

    // Write the network to a file
    if let Some(f) = file {
        //let path = Path::new(&f);
        solver.mut_network().save(f).unwrap();
    }
}

#[cfg(not(test))]
#[allow(unused_must_use)]
fn main() {
    env_logger::init();
    // Parse Arguments
    let args: Args = docopt::Docopt::new(MAIN_USAGE)
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());

    if args.cmd_train {
        #[cfg(all(feature = "cuda"))]
            train(
            args.flag_batchSize,
            args.flag_learningRate,
            args.flag_momentum,
            args.flag_file,
        );
        #[cfg(not(feature = "cuda"))]
        panic!(
            "
            Couldn't find CUDA, and Juice does not support RNNs on native CPU.
            If you have CUDA installed, and believe this is an error, please let us know on Gitter (https://gitter.im/spearow/juice)
            "
        )
    }
}
