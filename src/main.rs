extern crate rustc_serialize;
extern crate docopt;
extern crate csv;
extern crate tokio_core;
extern crate hyper;
extern crate futures;
extern crate log;

use std::io::prelude::*;
use std::fs::{OpenOptions,File};
use std::sync::{Arc, RwLock};

use hyper::Client;
use hyper::Uri;

extern crate hyper_tls;
use hyper_tls::HttpsConnector;

use std::str::FromStr;
use futures::Future;
use futures::Stream;

use docopt::Docopt;
use csv::Reader;


extern crate env_logger;
extern crate coaster as co;
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
    juice-examples (-h | --help)
    juice-examples --version


Options:
    <model-name>            Which MNIST model to use. Valid values: [linear, mlp, conv]

    -h --help               Show this screen.
    --version               Show version.
";

#[derive(Debug, RustcDecodable)]
struct MainArgs {
    arg_dataset_name: Option<String>,
    arg_model_name: Option<String>,
    arg_batch_size: Option<usize>,
    arg_learning_rate: Option<f32>,
    arg_momentum: Option<f32>,
    cmd_load_dataset: bool,
    cmd_mnist: bool,
    cmd_fashion: bool,
}

fn download_datasets(datasets: &[&str], base_url: &str) {
    for (i, v) in datasets.iter().enumerate() {
        println!("Downloading... {}/{}: {}", i + 1, datasets.len(), v);

        let uri = Uri::from_str(&format!("{}/{}", base_url, v)).unwrap();
        println!("URL: {}", &uri);

        let mut core = tokio_core::reactor::Core::new().unwrap();

        let handle = &core.handle();
        let response_fut = match uri.scheme() {
            Some("https") => {
                let client = Client::configure()
                    .connector(HttpsConnector::new(4, &handle).unwrap())
                    .build(&handle);
                client.get(uri)
            }

            Some("http") => Client::new(&handle).get(uri),

            _ => panic!("unsupported scheme"),
        };

        let work = response_fut.and_then(|res| {
            println!("Response: {}", res.status());

            let name = format!("assets/{}", v);
            {
                let _ = File::create(name.clone()).expect("Failed to create file");
            }
            res.body().for_each(move |chunk| {
                    let mut f = OpenOptions::new()
                        .append(true)
                        .open(name.clone())
                        .expect("Failed to open file in append mode");
                    f.write(&chunk)
                    .map(|_| ())
                    .map_err(From::from)
            })
        });
        core.run(work).unwrap();
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

        let mut decoder = GzDecoder::new(in_file.as_slice()).unwrap();

        decoder.read_to_end(&mut decompressed_file).unwrap();

        let filename_string = filename.split(".").nth(0).unwrap();

        File::create(format!("assets/{}", filename_string))
            .unwrap()
            .write_all(&decompressed_file as &[u8])
            .unwrap();
    }
}

#[cfg(not(test))]
#[allow(unused_must_use)]
fn main() {
    env_logger::init().unwrap();
    // Parse Arguments
    let args: MainArgs = Docopt::new(MAIN_USAGE)
        .and_then(|d| d.decode())
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
                // download_datasets(
                //     &datasets,
                //     "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com",
                // );
                // println!("{}", "Fashion MNIST dataset downloaded".to_string());
                // TODO avoid repeated effort here
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
    }
}

#[cfg(all(feature = "cuda"))]
fn run_mnist(
    model_name: Option<String>,
    batch_size: Option<usize>,
    learning_rate: Option<f32>,
    momentum: Option<f32>,
) {
    let example_count = 60000;
    let pixel_count = 784;
    let pixel_dim = 28;

    let mut rdr = Reader::from_file("assets/mnist_train.csv").unwrap();
    let mut decoded_images = rdr.decode().map(|row| {
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
            },
            _ => {
                println!("no value");
                panic!();
            }
    }
    });

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
                ReshapeConfig::of_shape(
                    &[batch_size, 1, pixel_dim, pixel_dim],
                ),
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
                LayerType::Reshape(
                    ReshapeConfig::of_shape(&[batch_size, pixel_count]),
                ),
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

    let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
        .base_path("./assets/")
        .label_format_digit()
        .training_set_length(example_count)
        .test_set_length(test_count)
        .finalize();

    let mut decoded_images = trn_img.chunks(pixel_count).enumerate().map(
        |(ind, pixels)| {
            (trn_lbl[ind], pixels.to_vec())
        },
    );

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
                ReshapeConfig::of_shape(
                    &[batch_size, 1, pixel_dim, pixel_dim],
                ),
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
                LayerType::Reshape(
                    ReshapeConfig::of_shape(&[batch_size, pixel_count]),
                ),
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
