#![allow(unused_must_use)]

use coaster as co;
use coaster_nn as conn;

use fs_err::File;
use juice::net::{LayerConfig, LinearConfig, Network, RnnConfig, SequentialConfig, WeightsData};
use juice::solver::RegressionLoss;
use juice::train::{OptimizerConfig, SgdWithMomentumConfig, Trainer, TrainerConfig};

use serde::Deserialize;
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};

#[cfg(all(feature = "cuda"))]
use co::frameworks::cuda::get_cuda_backend;
use co::prelude::*;
use conn::{DirectionMode, RnnInputMode, RnnNetworkMode};
use juice::util::*;

mod args;

use args::*;

pub(crate) const TRAIN_ROWS: usize = 35192;
pub(crate) const TEST_ROWS: usize = 8798;
pub(crate) const DATA_COLUMNS: usize = 10;

#[derive(Debug, Deserialize)]
struct Record {
    #[serde(rename = "Target")]
    target: f32,
    b1: f32,
    b2: f32,
    b3: f32,
    b4: f32,
    b5: f32,
    b6: f32,
    b7: f32,
    b8: f32,
    b9: f32,
    b10: f32,
}

impl Record {
    pub(crate) const fn target(&self) -> f32 {
        self.target
    }

    pub(crate) fn bs(&self) -> Vec<f32> {
        // only the b's
        vec![
            self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7, self.b8, self.b9,
            self.b10,
        ]
    }
}

// Provide an Iterator over the input data
pub(crate) fn data_generator(data: DataMode) -> impl Iterator<Item = (f32, Vec<f32>)> {
    let file = File::open(data.as_path()).expect("File opens as read. qed");
    let rdr = csv::ReaderBuilder::new()
        .delimiter(b',')
        .trim(csv::Trim::All)
        .from_reader(file);

    assert!(rdr.has_headers());

    rdr.into_deserialize()
        .enumerate()
        .map(move |(idx, row): (_, Result<Record, _>)| {
            let record: Record = match row {
                Ok(record) => record,
                Err(err) => panic!(
                    "All rows (including row {} (base-0)) in assets are valid. qed -> {:?}",
                    idx, err
                ),
            };
            (record.target(), record.bs())
        })
}

fn create_network() -> SequentialConfig {
    // Create a simple Network
    // * LSTM Layer
    // * Single Neuron
    // * Sigmoid Activation Function

    let mut net_cfg = SequentialConfig::default();

    net_cfg.add_layer(
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
    );

    net_cfg.add_layer("linear1", LinearConfig { output_size: 1 });
    net_cfg.add_layer("sigmoid", LayerConfig::Sigmoid);
    net_cfg
}

/// Train, and optionally, save the resulting network state/weights
pub(crate) fn train<Framework: IFramework + 'static>(
    backend: &Backend<Framework>,
    batch_size: usize,
    learning_rate: f32,
    momentum: f32,
    file: &PathBuf,
) where
    Backend<Framework>: coaster::IBackend + SolverOps<f32> + LayerOps<f32>,
{
    // Create the network.
    let net_cfg = create_network();
    let mut net = Network::from_config(backend, net_cfg, &[vec![DATA_COLUMNS, 1]]).unwrap();

    // Create the trainer with MSE objective function.
    let trainer_config = TrainerConfig {
        batch_size,
        objective: LayerConfig::MeanSquaredError,
        optimizer: OptimizerConfig::SgdWithMomentum(SgdWithMomentumConfig { momentum }),
        learning_rate,
        ..Default::default()
    };
    let mut trainer = Trainer::from_config(backend, trainer_config, &net, &vec![1]);

    // Define inputs & labels.
    let mut input = SharedTensor::<f32>::new(&[batch_size, DATA_COLUMNS, 1]);
    let mut label = SharedTensor::<f32>::new(&[batch_size, 1]);

    // Define Evaluation Method - Using Mean Squared Error
    let mut regression_evaluator =
        ::juice::solver::RegressionEvaluator::new(Some("mse".to_owned()));
    // Indicate how many samples to average loss over
    regression_evaluator.set_capacity(Some(2000));

    let mut data_rows = data_generator(DataMode::Train);
    let mut total = 0;
    for _ in 0..(TRAIN_ROWS / batch_size) {
        let mut targets = Vec::new();
        for (batch_n, (label_val, input_vals)) in data_rows.by_ref().take(batch_size).enumerate() {
            write_batch_sample(&mut input, &input_vals, batch_n);
            write_batch_sample(&mut label, &[label_val], batch_n);
            targets.push(label_val);
        }
        if targets.is_empty() {
            log::error!("Inconsistency detected - batch was empty");
            break;
        }

        total += targets.len();

        // Train the network.
        let mut inferred = trainer
            .train_minibatch(&backend, &mut net, &input, &label)
            .unwrap();
        let predictions = regression_evaluator.get_predictions(&mut inferred);
        regression_evaluator.add_samples(&predictions, &targets);
        println!(
            "Mean Squared Error {}",
            &regression_evaluator.accuracy() as &dyn RegressionLoss
        );
    }

    if total > 0 {
        let weights = net.copy_weights_data();
        File::create(file)
            .unwrap()
            .write_all(&serde_json::to_vec(&weights).unwrap())
            .unwrap();
    } else {
        panic!("No data was used for training");
    }
}

/// Test a the validation subset of data items against the trained network state.
pub(crate) fn test<Framework: IFramework + 'static>(
    backend: &Backend<Framework>,
    batch_size: usize,
    file_path: &Path,
) -> Result<(), Box<dyn std::error::Error>>
where
    Backend<Framework>: coaster::IBackend + SolverOps<f32> + LayerOps<f32>,
{
    // Load in a pre-trained network
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let weights: WeightsData = serde_json::from_reader(reader)?;
    let net_cfg = create_network();
    let net =
        Network::from_config_and_weights(backend, net_cfg, &[vec![DATA_COLUMNS, 1]], &weights)
            .unwrap();

    // Define Input & Labels
    let mut input = SharedTensor::<f32>::new(&[batch_size, DATA_COLUMNS, 1]);
    let mut label = SharedTensor::<f32>::new(&[batch_size, 1]);

    // Define Evaluation Method - Using Mean Squared Error
    let mut regression_evaluator =
        ::juice::solver::RegressionEvaluator::new(Some("mse".to_owned()));
    // Indicate how many samples to average loss over
    regression_evaluator.set_capacity(Some(2000));

    let mut data_rows = data_generator(DataMode::Test);

    for _ in 0..(TEST_ROWS / batch_size) {
        let mut targets = Vec::new();
        for (batch_n, (label_val, input_vals)) in data_rows.by_ref().take(batch_size).enumerate() {
            write_batch_sample(&mut input, &input_vals, batch_n);
            write_batch_sample(&mut label, &[label_val], batch_n);
            targets.push(label_val);
        }

        let mut results = net.transform(backend, &input)?;
        let predictions = regression_evaluator.get_predictions(&mut results);
        regression_evaluator.add_samples(&predictions, &targets);
        println!(
            "Mean Squared Error {}",
            &regression_evaluator.accuracy() as &dyn RegressionLoss
        );
    }
    Ok(())
}

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();
    // Parse Arguments
    let args: Args = docopt::Docopt::new(MAIN_USAGE)
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());

    #[cfg(all(feature = "cuda"))]
    {
        // Initialise a CUDA Backend, and the CUDNN and CUBLAS libraries.
        let backend = get_cuda_backend();

        match args.data_mode() {
            DataMode::Train => train(
                &backend,
                args.flag_batch_size.unwrap_or(default_batch_size()),
                args.flag_learning_rate.unwrap_or(default_learning_rate()),
                args.flag_momentum.unwrap_or(default_momentum()),
                &args.arg_networkfile,
            ),
            DataMode::Test => test(
                &backend,
                args.flag_batch_size.unwrap_or(default_batch_size()),
                &args.arg_networkfile,
            )
            .unwrap(),
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
        let args: Args = docopt
            .argv(args)
            .deserialize()
            .expect("Must deserialize. qed");

        assert_eq!(args, expected, "Expectations of {:?} stay unmet.", args);
    };

    check(
        &[
            "mackey-glass-example",
            "train",
            "--learning-rate=0.4",
            "--batch-size=11",
            "--momentum=0.17",
            "ahoi.capnp",
        ],
        Args {
            cmd_train: true,
            cmd_test: false,
            flag_batch_size: Some(11),
            flag_learning_rate: Some(0.4_f32),
            flag_momentum: Some(0.17_f32),
            arg_networkfile: PathBuf::from("ahoi.capnp"),
        },
    );

    check(
        &["mackey-glass-example", "test", "ahoi.capnp"],
        Args {
            cmd_train: false,
            cmd_test: true,
            flag_batch_size: None,
            flag_learning_rate: None,
            flag_momentum: None,
            arg_networkfile: PathBuf::from("ahoi.capnp"),
        },
    );
}

#[test]
fn test_data_is_ok() {
    assert_eq!(data_generator(DataMode::Train).count(), TRAIN_ROWS);
    assert_eq!(data_generator(DataMode::Test).count(), TEST_ROWS);
}
