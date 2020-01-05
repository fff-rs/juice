use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::sync::{Arc, RwLock};

use hyper::Client;
use hyper::Uri;

use hyper_rustls::HttpsConnector;

use std::str::FromStr;

use docopt::Docopt;

use csv::Reader;
use serde::Deserialize;

use tokio;

use futures_util::stream::TryStreamExt;

extern crate coaster as co;
extern crate env_logger;
extern crate juice;

use co::prelude::*;

use juice::layer::*;
use juice::layers::*;
use juice::solver::*;
use juice::util::*;

extern crate flate2;
use flate2::read::GzDecoder;

extern crate mnist;
use mnist::{Mnist, MnistBuilder};

const MAIN_USAGE: &'static str = "
Juice Examples

Usage:
    juice-examples load-dataset <dataset-name>
    juice-examples mnist <model-name> [--batch-size <batch-size>] [--learning-rate <learning-rate>] [--momentum <momentum>]
    juice-examples fashion <model-name> [--batch-size <batch-size>] [--learning-rate <learning-rate>] [--momentum <momentum>]
    juice-examples mackey-glass <model-name> [--batch-size <batch-size>] [--learning-rate <learning-rate>] [--momentum <momentum>]
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
    cmd_mackey_glass: bool,
}

async fn download_dataset(dataset: &str, base_url: &str) -> Result<(), hyper::Error> {
    let uri = Uri::from_str(&format!("{}/{}", base_url, dataset)).unwrap();
    println!("URL: {}", &uri);

    let res = match uri.scheme_str() {
        Some("https") => {
            let client: Client<_, hyper::Body> = Client::builder().build(HttpsConnector::new());

            client.get(uri)
        }

        Some("http") => Client::new().get(uri),

        _ => panic!("unsupported scheme"),
    }
    .await?;

    println!("Response: {}", res.status());

    let name = format!("assets/{}", dataset);
    {
        let _ = File::create(name.clone()).expect("Failed to create file");
    }

    let mut x = res.into_body();
    while let Some(chunk) = x.try_next().await? {
        let mut f = OpenOptions::new()
            .append(true)
            .open(name.clone())
            .expect("Failed to open file in append mode");

        f.write(&chunk).unwrap();
    }
    Ok(())
}

fn download_datasets(datasets: &[&str], base_url: &str) {
    let mut rt = tokio::runtime::Runtime::new().unwrap();
    for (i, v) in datasets.iter().enumerate() {
        println!("Downloading... {}/{}: {}", i + 1, datasets.len(), v);
        let _ = rt.block_on(async { download_dataset(*v, base_url) });
    }
}

fn unzip_datasets(datasets: &[&str]) {
    for filename in datasets {
        println!("Decompressing: {}", filename);

        // TODO figure out how to specify asset dir
        let mut file_handle = File::open(&format!("./assets/{}", filename)).unwrap();
        let mut in_file: Vec<u8> = Vec::new();
        let mut decompressed_file: Vec<u8> = Vec::new();

        file_handle.read_to_end(&mut in_file).unwrap();

        let mut decoder = GzDecoder::new(in_file.as_slice());

        decoder.read_to_end(&mut decompressed_file).unwrap();

        let filename_string = filename.split(".").nth(0).unwrap();

        File::create(format!("assets/{}", filename_string))
            .unwrap()
            .write_all(&decompressed_file as &[u8])
            .unwrap();
    }
}

use serde;

#[cfg(not(test))]
#[allow(unused_must_use)]
fn main() {
    env_logger::init();
    // Parse Arguments
    let args: Args = Docopt::new(MAIN_USAGE)
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());

    if args.cmd_load_dataset {
        match &*args.arg_dataset_name.unwrap() {
            "mnist" => {
                let datasets = ["mnist_test.csv", "mnist_train.csv"];
                download_datasets(&datasets, "https://pjreddie.com/media/files");
                println!("{}", "MNIST dataset downloaded".to_string())
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
                    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com",
                );
                println!("{}", "Fashion MNIST dataset downloaded".to_string());

                unzip_datasets(&datasets);
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
    } else if args.cmd_mackey_glass {
        #[cfg(all(feature = "cuda"))]
        run_mackey_glass(
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

fn get_mnist_iter(pixel_count: usize) -> impl Iterator<Item = (u8, Vec<u8>)> {
    let rdr = Reader::from_reader(File::open("assets/mnist_train.csv").unwrap());

    rdr.into_deserialize().map(move |row| {
        match row {
            Ok(value) => {
                let row_vec: Box<Vec<u8>> = Box::new(value);
                let label = row_vec[0];
                let mut pixels = vec![0u8; pixel_count];
                for (place, element) in pixels.iter_mut().zip(row_vec.iter().skip(1)) {
                    *place = *element;
                }
                // TODO: reintroduce Coaster
                // let img = Image::from_luma_pixels(pixel_dim, pixel_dim, pixels);
                // match img {
                //     Ok(in_img) => {
                //         println!("({}): {:?}", label, in_img.transform(vec![pixel_dim, pixel_dim]));
                //     },
                //     Err(_) => unimplemented!()
                // }
                (label, pixels)
            }
            _ => {
                println!("no value");
                panic!();
            }
        }
    })
}

#[cfg(all(feature = "cuda"))]
#[allow(dead_code)]
fn run_mnist(
    model_name: Option<String>,
    batch_size: Option<usize>,
    learning_rate: Option<f32>,
    momentum: Option<f32>,
) {
    let example_count = 60000;
    let pixel_count = 784;
    let pixel_dim = 28;

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

    // set up backends
    let backend = ::std::rc::Rc::new(Backend::<Cuda>::default().unwrap());
    // let native_backend = ::std::rc::Rc::new(Backend::<Native>::default().unwrap());

    // set up solver
    let mut solver_cfg = SolverConfig {
        minibatch_size: batch_size,
        base_lr: learning_rate,
        momentum: momentum,
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

    let mut decoded_images = get_mnist_iter(pixel_count);
    for _ in 0..(example_count / batch_size) {
        // write input
        let mut targets = Vec::new();
        for (batch_n, (label_val, input)) in decoded_images.by_ref().take(batch_size).enumerate() {
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

        println!("Accuracy {}", confusion.accuracy());
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
        .base_path("./assets/")
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

    // set up backends
    let backend = ::std::rc::Rc::new(Backend::<Cuda>::default().unwrap());
    // let native_backend = ::std::rc::Rc::new(Backend::<Native>::default().unwrap());

    // set up solver
    let mut solver_cfg = SolverConfig {
        minibatch_size: batch_size,
        base_lr: learning_rate,
        momentum: momentum,
        ..SolverConfig::default()
    };
    solver_cfg.network = LayerConfig::new("network", net_cfg);
    solver_cfg.objective = LayerConfig::new("classifier", classifier_cfg);
    let mut solver = Solver::from_config(backend.clone(), backend.clone(), &solver_cfg);

    // set up confusion matrix
    let mut confusion = ::juice::solver::ConfusionMatrix::new(10);
    confusion.set_capacity(Some(1000));

    let inp = SharedTensor::<f32>::new(&[batch_size, pixel_dim, pixel_dim]);
    let label = SharedTensor::<f32>::new(&[batch_size, 1]);

    let inp_lock = Arc::new(RwLock::new(inp));
    let label_lock = Arc::new(RwLock::new(label));

    for _ in 0..(example_count / batch_size as u32) {
        // write input
        let mut targets = Vec::new();

        for (batch_n, (label_val, ref input)) in
            decoded_images.by_ref().take(batch_size).enumerate()
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
#[allow(dead_code)]
fn run_mackey_glass(
    model_name: Option<String>,
    batch_size: Option<usize>,
    learning_rate: Option<f32>,
    momentum: Option<f32>,
) {
    let example_count: usize = 11751;
    let columns: usize = 10;

    let batch_size = batch_size.unwrap_or(65);
    let learning_rate = learning_rate.unwrap_or(0.03f32);
    let momentum = momentum.unwrap_or(0f32);

    let mut net_cfg = SequentialConfig::default();
    net_cfg.add_input("data", &[batch_size, 1_usize, columns]);
    net_cfg.force_backward = true;

    match &*model_name.unwrap_or("none".to_owned()) {
        "linear" => {
            net_cfg.add_layer(LayerConfig::new(
                "linear1",
                LinearConfig { output_size: 50 },
            ));
            net_cfg.add_layer(LayerConfig::new(
                "linear2",
                LinearConfig { output_size: 10 },
            ));
            net_cfg.add_layer(LayerConfig::new(
                "linear3",
                LinearConfig { output_size: 1 },
            ));
            net_cfg.add_layer(LayerConfig::new(
                "sigmoid",
                LayerType::Sigmoid
            ));
        },
        "lstm-dense" => {
            net_cfg.add_layer(LayerConfig::new(
                "LSTMInitial",
                RnnConfig { output_size: 10, cell_size: 10, hidden_size: 10, num_layers: 10, rnn_type: RnnType::LSTM }
            ));
            net_cfg.add_layer(LayerConfig::new(
                "linear1",
                LinearConfig { output_size: 1}
            ));
            net_cfg.add_layer(LayerConfig::new(
                "sigmoid",
                LayerType::Sigmoid
            ));
        }
        _ => panic!("Only linear & lstm-dense models are currently implemented for mackey-glass"),
    }

    let mut regressor_cfg = SequentialConfig::default();
    regressor_cfg.add_input("network_out", &[batch_size, 1]);
    regressor_cfg.add_input("label", &[batch_size, 1]);
    // set up mse loss
    let mse_layer_cfg = LayerConfig::new("mse", LayerType::MeanSquaredError);
    regressor_cfg.add_layer(mse_layer_cfg);

    // set up backends
    let backend = ::std::rc::Rc::new(Backend::<Cuda>::default().unwrap());

    // set up solver
    let mut solver_cfg = SolverConfig {
        minibatch_size: batch_size,
        base_lr: learning_rate,
        momentum,
        ..SolverConfig::default()
    };
    solver_cfg.network = LayerConfig::new("network", net_cfg);
    solver_cfg.objective = LayerConfig::new("regressor", regressor_cfg);
    let mut solver = Solver::from_config(backend.clone(), backend.clone(), &solver_cfg);

    let inp = SharedTensor::<f32>::new(&[batch_size, 1, columns]);
    let label = SharedTensor::<f32>::new(&[batch_size, 1]);

    let inp_lock = Arc::new(RwLock::new(inp));
    let label_lock = Arc::new(RwLock::new(label));

    // set up evaluator for regression
    let mut regr_eval = ::juice::solver::RegressionEvaluator::new(Some("mse".to_owned()));
    regr_eval.set_capacity(Some(500));

    let mut data_rows = get_regr_iter();
    for _ in 0..(example_count / batch_size) {
        // write input
        let mut targets = Vec::new();
        for (batch_n, (label_val, input)) in data_rows.by_ref().take(batch_size).enumerate() {
            let mut inp = inp_lock.write().unwrap();
            let mut label = label_lock.write().unwrap();
            write_batch_sample(&mut inp, &input, batch_n);
            write_batch_sample(&mut label, &[label_val], batch_n);
            targets.push(label_val);
        }
        // train the network!
        let inferred_out = solver.train_minibatch(inp_lock.clone(), label_lock.clone());

        let mut inferred = inferred_out.write().unwrap();
        let predictions = regr_eval.get_predictions(&mut inferred);
        regr_eval.add_samples(&predictions, &targets);
        println!(
            "Mean Squared Error {}",
            &regr_eval.accuracy() as &dyn RegressionLoss
        );
    }
}

fn get_regr_iter() -> impl Iterator<Item = (f32, Vec<f32>)> {
    let rdr = Reader::from_reader(File::open("assets/normalised_mackeyglass.csv").unwrap());
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

fn get_packed_regr_iter() -> impl Iterator<Item = (f32, Vec<Vec<f32>>)> {
    let rdr = Reader::from_reader(File::open("assets/normalised_mackeyglass_lstm.csv").unwrap());
    let columns: usize = 10;

    rdr
        .into_deserialize()
        .map( move | row| {
            match row {
                Ok(value) => {
                    let row_vec: Box <Vec<f32>> = Box::new(value);
                    let label = row_vec[0];
                    let columns = row_vec[1..=columns].to_vec();
                    (label, vec![columns])
                },
                _ => {
                    println ! ("no value");
                    panic ! ();
                }
            }
        })
}
