extern crate coaster as co;
extern crate coaster_nn as conn;
extern crate env_logger;
extern crate juice;
extern crate juice_utils;
extern crate mnist;

#[cfg(all(feature = "cuda"))]
use co::frameworks::cuda::get_cuda_backend;
use co::prelude::*;
use juice::layer::*;
use juice::layers::*;
use juice::solver::*;
use juice::util::*;
use juice_utils::{download_datasets, unzip_datasets};
use mnist::{Mnist, MnistBuilder};
use serde::Deserialize;
use std::rc::Rc;
use std::sync::{Arc, RwLock};


const MAIN_USAGE: &str = "
Juice Examples

Usage:
    juice-examples load-dataset <dataset-name>
    juice-examples mnist <model-name> [--batch-size <batch-size>] [--learning-rate <learning-rate>] [--momentum <momentum>]
    juice-examples fashion <model-name> [--batch-size <batch-size>] [--learning-rate <learning-rate>] [--momentum <momentum>]
    juice-examples (-h | --help)
    juice-examples --version


Options:
    <model-name>            Which MNIST model to use. Valid values: [linear, mlp, conv]
    <dataset-name>          Which dataset to download. Valid values: [mnist, fashion]

    -h --help               Show this screen.
    --version               Show version.
";

#[derive(Debug, Deserialize)]
struct Args {
    arg_dataset_name: Option<String>,
    arg_model_name: Option<String>,
    arg_batch_size: Option<usize>,
    arg_learning_rate: Option<f32>,
    arg_momentum: Option<f32>,
    cmd_load_dataset: bool,
    cmd_mnist: bool,
    cmd_fashion: bool,
}

#[cfg(not(test))]
#[allow(unused_must_use)]
fn main() {
    env_logger::init();
    // Parse Arguments
    let args: Args = docopt::Docopt::new(MAIN_USAGE)
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());

    if args.cmd_load_dataset {
        match &*args.arg_dataset_name.unwrap() {
            "mnist" => {
                let datasets = [
                    "train-images-idx3-ubyte.gz",
                    "train-labels-idx1-ubyte.gz",
                    "t10k-images-idx3-ubyte.gz",
                    "t10k-labels-idx1-ubyte.gz"
                ];
                download_datasets(
                    &datasets,
                    &"./assets/mnist/",
                    "http://yann.lecun.com/exdb/mnist/").unwrap();

                unzip_datasets(&datasets, &"./assets/mnist/");
            }

            "fashion" => {
                let datasets = [
                    "train-images-idx3-ubyte.gz",
                    "train-labels-idx1-ubyte.gz",
                    "t10k-images-idx3-ubyte.gz",
                    "t10k-labels-idx1-ubyte.gz",
                ];

                download_datasets(
                    &datasets,
                    &"./assets/mnist-fashion/",
                    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com",
                ).unwrap();
                println!("{}", "Fashion MNIST dataset downloaded".to_string());

                unzip_datasets(&datasets, &"./assets/mnist-fashion/");
                println!("{}", "Fashion MNIST dataset decompressed".to_string());
            }
            _ => println!("{}", "Failed to download MNIST dataset!".to_string()),
        }
    } else if args.cmd_mnist {
        #[cfg(all(feature = "cuda"))]
            run_mnist(
            args.arg_model_name,
            args.arg_batch_size,
            args.arg_learning_rate,
            args.arg_momentum,
        );
        #[cfg(not(feature = "cuda"))]
            {
                println!(
                    "Right now, you really need cuda! Not all features are available for all backends and as such, this one -as of now - only works with cuda."
                );
                panic!()
            }
    } else if args.cmd_fashion {
        #[cfg(all(feature = "cuda"))]
            run_fashion(
            args.arg_model_name,
            args.arg_batch_size,
            args.arg_learning_rate,
            args.arg_momentum,
        );
        #[cfg(not(feature = "cuda"))]
            {
                println!(
                    "Right now, you really need cuda! Not all features are available for all backends and as such, this one -as of now - only works with cuda."
                );
                panic!()
            }
    }
}

#[allow(dead_code)]
fn run_mnist(
    model_name: Option<String>,
    batch_size: Option<usize>,
    learning_rate: Option<f32>,
    momentum: Option<f32>,
) {
    let example_count = 60000;
    let test_count = 10000;
    let pixel_count = 784;
    let pixel_dim = 28;

    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .base_path("./assets/mnist/")
        .label_format_digit()
        .training_set_length(example_count)
        .test_set_length(test_count)
        .finalize();

    let mut decoded_images = trn_img
        .chunks(pixel_count)
        .enumerate()
        .map(|(ind, pixels)| (trn_lbl[ind], pixels.to_vec()));

    let batch_size = batch_size.unwrap_or(1);
    let learning_rate = learning_rate.unwrap_or(0.001f32);
    let momentum = momentum.unwrap_or(0f32);

    let mut net_cfg = SequentialConfig::default();
    net_cfg.add_input("data", &[batch_size, pixel_dim, pixel_dim]);
    net_cfg.force_backward = true;

    match &*model_name.unwrap_or("none".to_owned()) {
        "conv" => {
            net_cfg.add_layer(LayerConfig::new(
                "reshape",
                ReshapeConfig::of_shape(&[batch_size, 1, pixel_dim, pixel_dim]),
            ));
            net_cfg.add_layer(LayerConfig::new(
                "conv",
                ConvolutionConfig {
                    num_output: 20,
                    filter_shape: vec![5],
                    padding: vec![0],
                    stride: vec![1],
                },
            ));
            net_cfg.add_layer(LayerConfig::new(
                "pooling",
                PoolingConfig {
                    mode: PoolingMode::Max,
                    filter_shape: vec![2],
                    padding: vec![0],
                    stride: vec![2],
                },
            ));
            net_cfg.add_layer(LayerConfig::new(
                "linear1",
                LinearConfig { output_size: 500 },
            ));
            net_cfg.add_layer(LayerConfig::new("sigmoid", LayerType::Sigmoid));
            net_cfg.add_layer(LayerConfig::new(
                "linear2",
                LinearConfig { output_size: 10 },
            ));
        }
        "mlp" => {
            net_cfg.add_layer(LayerConfig::new(
                "reshape",
                LayerType::Reshape(ReshapeConfig::of_shape(&[batch_size, pixel_count])),
            ));
            net_cfg.add_layer(LayerConfig::new(
                "linear1",
                LayerType::Linear(LinearConfig { output_size: 1568 }),
            ));
            net_cfg.add_layer(LayerConfig::new("sigmoid", LayerType::Sigmoid));
            net_cfg.add_layer(LayerConfig::new(
                "linear2",
                LayerType::Linear(LinearConfig { output_size: 10 }),
            ));
        }
        "linear" => {
            net_cfg.add_layer(LayerConfig::new(
                "linear",
                LayerType::Linear(LinearConfig { output_size: 10 }),
            ));
        }
        _ => panic!("Unknown model. Try one of [linear, mlp, conv]"),
    }
    net_cfg.add_layer(LayerConfig::new("log_softmax", LayerType::LogSoftmax));

    let mut classifier_cfg = SequentialConfig::default();
    classifier_cfg.add_input("network_out", &[batch_size, 10]);
    classifier_cfg.add_input("label", &[batch_size, 1]);
    // set up nll loss
    let nll_layer_cfg = NegativeLogLikelihoodConfig { num_classes: 10 };
    let nll_cfg = LayerConfig::new("nll", LayerType::NegativeLogLikelihood(nll_layer_cfg));
    classifier_cfg.add_layer(nll_cfg);

    #[cfg(all(feature = "cuda"))]
        let backend = Rc::new(get_cuda_backend());
    #[cfg(not(feature = "cuda"))]
        let backend = Rc::new(get_native_backend());
    // set up solver
    let mut solver_cfg = SolverConfig {
        minibatch_size: batch_size,
        base_lr: learning_rate,
        momentum,
        ..SolverConfig::default()
    };
    solver_cfg.network = LayerConfig::new("network", net_cfg);
    solver_cfg.objective = LayerConfig::new("classifier", classifier_cfg);
    let mut solver = Solver::from_config(backend.clone(), backend.clone(), &solver_cfg);

    let inp = SharedTensor::<f32>::new(&[batch_size, pixel_dim, pixel_dim]);
    let label = SharedTensor::<f32>::new(&[batch_size, 1]);

    let inp_lock = Arc::new(RwLock::new(inp));
    let label_lock = Arc::new(RwLock::new(label));

    // set up confusion matrix
    let mut confusion = ::juice::solver::ConfusionMatrix::new(10);
    confusion.set_capacity(Some(1000));

    for _ in 0..(example_count / batch_size as u32) {
        // write input
        let mut targets = Vec::new();

        for (batch_n, (label_val, ref input)) in decoded_images
            .by_ref()
            .take(batch_size)
            .enumerate()
        {
            let mut inp = inp_lock.write().unwrap();
            let mut label = label_lock.write().unwrap();
            write_batch_sample(&mut inp, &input, batch_n);
            write_batch_sample(&mut label, &[label_val], batch_n);

            targets.push(label_val as usize);
        }
        // train the network!
        let infered_out = solver.train_minibatch(inp_lock.clone(), label_lock.clone());

        let mut infered = infered_out.write().unwrap();
        let predictions = confusion.get_predictions(&mut infered);

        confusion.add_samples(&predictions, &targets);
        println!(
            "Last sample: {} | Accuracy {}",
            confusion.samples().iter().last().unwrap(),
            confusion.accuracy()
        );
    }
}

#[cfg(all(feature = "cuda"))]
fn run_fashion(
    model_name: Option<String>,
    batch_size: Option<usize>,
    learning_rate: Option<f32>,
    momentum: Option<f32>,
) {
    let example_count = 60000;
    let test_count = 10000;
    let pixel_count = 784;
    let pixel_dim = 28;

    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .base_path("./assets/mnist-fashion/")
        .label_format_digit()
        .training_set_length(example_count)
        .test_set_length(test_count)
        .finalize();

    let mut decoded_images = trn_img
        .chunks(pixel_count)
        .enumerate()
        .map(|(ind, pixels)| (trn_lbl[ind], pixels.to_vec()));

    let batch_size = batch_size.unwrap_or(30);
    let learning_rate = learning_rate.unwrap_or(0.001f32);
    let momentum = momentum.unwrap_or(0f32);

    let mut net_cfg = SequentialConfig::default();
    net_cfg.add_input("data", &[batch_size, pixel_dim, pixel_dim]);
    net_cfg.force_backward = true;

    match &*model_name.unwrap_or("none".to_owned()) {
        "conv" => {
            net_cfg.add_layer(LayerConfig::new(
                "reshape",
                ReshapeConfig::of_shape(&[batch_size, 1, pixel_dim, pixel_dim]),
            ));
            net_cfg.add_layer(LayerConfig::new(
                "conv",
                ConvolutionConfig {
                    num_output: 20,
                    filter_shape: vec![5],
                    padding: vec![0],
                    stride: vec![1],
                },
            ));
            net_cfg.add_layer(LayerConfig::new(
                "pooling",
                PoolingConfig {
                    mode: PoolingMode::Max,
                    filter_shape: vec![2],
                    padding: vec![0],
                    stride: vec![2],
                },
            ));
            net_cfg.add_layer(LayerConfig::new(
                "linear1",
                LinearConfig { output_size: 500 },
            ));
            net_cfg.add_layer(LayerConfig::new("sigmoid", LayerType::Sigmoid));
            net_cfg.add_layer(LayerConfig::new(
                "linear2",
                LinearConfig { output_size: 10 },
            ));
        }
        "mlp" => {
            net_cfg.add_layer(LayerConfig::new(
                "reshape",
                LayerType::Reshape(ReshapeConfig::of_shape(&[batch_size, pixel_count])),
            ));
            net_cfg.add_layer(LayerConfig::new(
                "linear1",
                LayerType::Linear(LinearConfig { output_size: 1568 }),
            ));
            net_cfg.add_layer(LayerConfig::new("sigmoid", LayerType::Sigmoid));
            net_cfg.add_layer(LayerConfig::new(
                "linear2",
                LayerType::Linear(LinearConfig { output_size: 10 }),
            ));
        }
        "linear" => {
            net_cfg.add_layer(LayerConfig::new(
                "linear",
                LayerType::Linear(LinearConfig { output_size: 10 }),
            ));
        }
        _ => panic!("Unknown model. Try one of [linear, mlp, conv]"),
    }
    net_cfg.add_layer(LayerConfig::new("log_softmax", LayerType::LogSoftmax));

    let mut classifier_cfg = SequentialConfig::default();
    classifier_cfg.add_input("network_out", &[batch_size, 10]);
    classifier_cfg.add_input("label", &[batch_size, 1]);
    // set up nll loss
    let nll_layer_cfg = NegativeLogLikelihoodConfig { num_classes: 10 };
    let nll_cfg = LayerConfig::new("nll", LayerType::NegativeLogLikelihood(nll_layer_cfg));
    classifier_cfg.add_layer(nll_cfg);

    // set up backends
    let backend = Rc::new(get_cuda_backend());

    // set up solver
    let mut solver_cfg = SolverConfig {
        minibatch_size: batch_size,
        base_lr: learning_rate,
        momentum,
        ..SolverConfig::default()
    };
    solver_cfg.network = LayerConfig::new("network", net_cfg);
    solver_cfg.objective = LayerConfig::new("classifier", classifier_cfg);
    let mut solver = Solver::from_config(backend.clone(), backend.clone(), &solver_cfg);

    // set up confusion matrix
    let mut classification_evaluator = ::juice::solver::ConfusionMatrix::new(10);
    classification_evaluator.set_capacity(Some(1000));

    let input = SharedTensor::<f32>::new(&[batch_size, pixel_dim, pixel_dim]);
    let inp_lock = Arc::new(RwLock::new(input));

    let label = SharedTensor::<f32>::new(&[batch_size, 1]);
    let label_lock = Arc::new(RwLock::new(label));

    for _ in 0..(example_count / batch_size as u32) {
        // write input
        let mut targets = Vec::new();

        for (batch_n, (label_val, ref input)) in
        decoded_images.by_ref().take(batch_size).enumerate()
        {
            let mut input_tensor = inp_lock.write().unwrap();
            let mut label_tensor = label_lock.write().unwrap();
            write_batch_sample(&mut input_tensor, &input, batch_n);
            write_batch_sample(&mut label_tensor, &[label_val], batch_n);
            targets.push(label_val as usize);
        }
        // train the network!
        let infered_out = solver.train_minibatch(inp_lock.clone(), label_lock.clone());

        let mut infered = infered_out.write().unwrap();
        let predictions = classification_evaluator.get_predictions(&mut infered);

        classification_evaluator.add_samples(&predictions, &targets);
        println!(
            "Last sample: {} | Accuracy {}",
            classification_evaluator.samples().iter().last().unwrap(),
            classification_evaluator.accuracy()
        );
    }
}