extern crate coaster as co;
extern crate coaster_nn as conn;
extern crate env_logger;
extern crate juice;
extern crate juice_utils;
extern crate mnist;

#[cfg(feature = "cuda")]
use co::frameworks::cuda::get_cuda_backend;
#[cfg(not(feature = "cuda"))]
use co::frameworks::native::get_native_backend;
use co::prelude::*;
use juice::net::*;
use juice::train::*;
use juice::util::*;
use juice_utils::{download_datasets, unzip_datasets};
use mnist::{Mnist, MnistBuilder};
use serde::Deserialize;

// TODO: Add a choice for the optimizer (SGD or Adam).
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

enum MnistType {
    Fashion,
    Numbers,
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
                    "t10k-labels-idx1-ubyte.gz",
                ];
                download_datasets(
                    &datasets,
                    &"./assets/mnist/",
                    "http://yann.lecun.com/exdb/mnist/",
                )
                .unwrap();

                unzip_datasets(&datasets, &"./assets/mnist/").unwrap();
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
                )
                .unwrap();
                println!("{}", "Fashion MNIST dataset downloaded".to_string());

                unzip_datasets(&datasets, &"./assets/mnist-fashion/").unwrap();
                println!("{}", "Fashion MNIST dataset decompressed".to_string());
            }
            _ => println!("{}", "Failed to download MNIST dataset!".to_string()),
        }
    } else if args.cmd_mnist {
        #[cfg(all(feature = "cuda"))]
        run_mnist(
            MnistType::Numbers,
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
        run_mnist(
            MnistType::Fashion,
            args.arg_model_name,
            args.arg_batch_size,
            args.arg_learning_rate,
            args.arg_momentum,
        );
    }
}

#[cfg(all(feature = "cuda"))]
fn add_conv_net(mut net_cfg: SequentialConfig) -> SequentialConfig {
    net_cfg.add_layer(
        "conv",
        ConvolutionConfig {
            feature_maps: 20,
            kernel_size: 5,
            padding: 0,
            stride: 1,
        },
    );
    net_cfg.add_layer(
        "pooling",
        PoolingConfig {
            mode: PoolingMode::Max,
            window_size: 2,
            padding: 0,
            stride: 2,
        },
    );
    net_cfg.add_layer("linear1", LinearConfig { output_size: 500 });
    net_cfg.add_layer("sigmoid", LayerConfig::Sigmoid);
    net_cfg.add_layer("linear2", LinearConfig { output_size: 10 });
    net_cfg
}

#[cfg(not(feature = "cuda"))]
fn add_conv_net(_net_cfg: SequentialConfig) -> SequentialConfig {
    println!(
        "Currently Juice does not have a native pooling function to use with Conv Nets - you can either try
        the CUDA implementation, or use a different type of layer"
    );
    panic!()
}

fn add_mlp(mut net_cfg: SequentialConfig) -> SequentialConfig {
    net_cfg.add_layer("linear1", LinearConfig { output_size: 1568 });
    net_cfg.add_layer("sigmoid", LayerConfig::Sigmoid);
    net_cfg.add_layer("linear2", LinearConfig { output_size: 10 });
    net_cfg
}

fn add_linear_net(mut net_cfg: SequentialConfig) -> SequentialConfig {
    net_cfg.add_layer("linear", LinearConfig { output_size: 10 });
    net_cfg
}

fn run_mnist(
    mnist_type: MnistType,
    model_name: Option<String>,
    batch_size: Option<usize>,
    learning_rate: Option<f32>,
    momentum: Option<f32>,
) {
    let example_count = 60000;
    let test_count = 10000;
    let pixel_count = 784;
    let pixel_dim = 28;

    let asset_path = match mnist_type {
        MnistType::Fashion => "./assets/mnist-fashion",
        MnistType::Numbers => "./assets/mnist",
    };

    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .base_path(asset_path)
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
    let momentum = momentum.unwrap_or(0.1f32);

    // Create the backend.
    #[cfg(all(feature = "cuda"))]
    let backend = get_cuda_backend();
    #[cfg(not(feature = "cuda"))]
    let backend = Rc::new(get_native_backend());

    // Create the network configuration and the net itself.
    let mut net_cfg = SequentialConfig::default();
    net_cfg = match &*model_name.unwrap_or("none".to_owned()) {
        "conv" => add_conv_net(net_cfg),
        "mlp" => add_mlp(net_cfg),
        "linear" => add_linear_net(net_cfg),
        _ => panic!("Unknown model. Try one of [linear, mlp, conv]"),
    };
    net_cfg.add_layer("log_softmax", LayerConfig::LogSoftmax);
    let mut net =
        Network::from_config(&backend, net_cfg, &[vec![1, pixel_dim, pixel_dim]]).unwrap();

    // Create the trainer.
    let trainer_config = TrainerConfig {
        batch_size,
        objective: LayerConfig::NegativeLogLikelihood,
        optimizer: OptimizerConfig::SgdWithMomentum(SgdWithMomentumConfig { momentum }),
        learning_rate,
        ..Default::default()
    };
    let mut trainer = Trainer::from_config(&backend, trainer_config, &net, &vec![1]);

    // Set up confusion matrix.
    let mut classification_evaluator = ::juice::solver::ConfusionMatrix::new(10);
    classification_evaluator.set_capacity(Some(1000));

    let mut input = SharedTensor::<f32>::new(&[batch_size, pixel_dim, pixel_dim]);
    let mut label = SharedTensor::<f32>::new(&[batch_size, 1]);

    for _ in 0..(example_count / batch_size as u32) {
        // write input
        let mut targets = Vec::new();

        for (batch_n, (label_val, ref input_bytes)) in
            decoded_images.by_ref().take(batch_size).enumerate()
        {
            write_batch_sample(&mut input, &input_bytes, batch_n);
            write_batch_sample(&mut label, &[label_val], batch_n);
            targets.push(label_val as usize);
        }
        // train the network!
        let mut infered = trainer.train_minibatch(&backend, &mut net, &input, &label);

        let predictions = classification_evaluator.get_predictions(&mut infered);

        classification_evaluator.add_samples(&predictions, &targets);
        println!(
            "Last sample: {} | Accuracy {}",
            classification_evaluator.samples().iter().last().unwrap(),
            classification_evaluator.accuracy()
        );
    }
}
