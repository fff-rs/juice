extern crate rustc_serialize;
extern crate docopt;
extern crate csv;
extern crate hyper;
extern crate leaf;
extern crate collenchyma as co;
extern crate collenchyma_blas as coblas;

use std::io::prelude::*;
use std::fs::File;
use std::sync::{Arc, RwLock};

use hyper::Client;

use docopt::Docopt;
use csv::{Reader};
use leaf::network::{Network, NetworkConfig};
use leaf::layer::{LayerConfig, LayerType};
use leaf::layers::*;
use co::prelude::*;
use coblas::plugin::Copy;

const MAIN_USAGE: &'static str = "
Leaf Examples

Usage:
    leaf-examples load-dataset <dataset-name>
    leaf-examples mnist <model-name> [--batch-size <batch-size>] [--learning-rate <learning-rate>]
    leaf-examples (-h | --help)
    leaf-examples --version

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
    cmd_load_dataset: bool,
    cmd_mnist: bool,
}

#[cfg(not(test))]
#[allow(unused_must_use)]
fn main() {
    // Parse Arguments
    let args: MainArgs = Docopt::new(MAIN_USAGE)
                            .and_then(|d| d.decode())
                            .unwrap_or_else(|e| e.exit());

    if args.cmd_load_dataset {
        match &*args.arg_dataset_name.unwrap() {
            "mnist" => {
                let datasets = ["mnist_test.csv", "mnist_train.csv"];
                for (i, v) in datasets.iter().enumerate() {
                    println!("Downloading... {}/{}: {}", i+1, datasets.len(), v);
                    let mut body = String::new();

                    Client::new()
                        .get(&format!("http://pjreddie.com/media/files/{}", v))
                        .send().unwrap().read_to_string(&mut body);

                    File::create(format!("assets/{}", v))
                        .unwrap()
                        .write_all(&body.into_bytes());
                }
                println!("{}", "awesome".to_string())
            },
            _ => println!("{}", "fail".to_string())
        }
    } else if args.cmd_mnist {
        run_mnist(args.arg_model_name, args.arg_batch_size, args.arg_learning_rate);
    }
}

#[allow(dead_code)]
fn run_mnist(model_name: Option<String>, batch_size: Option<usize>, learning_rate: Option<f32>) {
    let mut rdr = Reader::from_file("assets/mnist_train.csv").unwrap();
    let mut decoded_images = rdr.decode().map(|row|
        match row {
            Ok(value) => {
                let row_vec: Box<Vec<u8>> = Box::new(value);
                let label = row_vec[0];
                let mut pixels = vec![0u8; 784];
                for (place, element) in pixels.iter_mut().zip(row_vec.iter().skip(1)) {
                    *place = *element;
                }
                // TODO: reintroduce Cuticula
                // let img = Image::from_luma_pixels(28, 28, pixels);
                // match img {
                //     Ok(in_img) => {
                //         println!("({}): {:?}", label, in_img.transform(vec![28, 28]));
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
    );

    let batch_size = batch_size.unwrap_or(1);
    let learning_rate = learning_rate.unwrap_or(0.001f32);
    let mut net_cfg = NetworkConfig::default();
    net_cfg.name = "Simple MNIST".to_owned();
    // set up inputs
    net_cfg.add_input("data", &vec![batch_size, 1, 28, 28]);
    net_cfg.add_input("label", &vec![batch_size, 1]);
    net_cfg.force_backward = true;

    match &*model_name.unwrap_or("none".to_owned()) {
        "conv" => {
            // set up reshape layer
            let mut reshape_cfg = LayerConfig::new("reshape", LayerType::Reshape(ReshapeConfig::of_shape(&vec![batch_size, 1, 28, 28])));
            reshape_cfg.add_input("data"); // provide the layer we want to reshape as bottom
            // reshape_cfg.add_output("data"); // provide the layer we want to reshape as bottom
            net_cfg.add_layer(reshape_cfg);
            // set up conv layer
            let conv_cfg = ConvolutionConfig { num_output: 20, filter_shape: vec![5], stride: vec![1], padding: vec![0], axis: None };
            let mut conv_cfg = LayerConfig::new("conv", LayerType::Convolution(conv_cfg));
            conv_cfg.add_input("data");
            conv_cfg.add_output("conv_out");
            net_cfg.add_layer(conv_cfg);
            // set up pooling layer
            let pool_cfg = PoolingConfig { mode: PoolingMode::Max, filter_shape: vec![2], stride: vec![2], padding: vec![0] };
            let mut pool_cfg = LayerConfig::new("pooling", LayerType::Pooling(pool_cfg));
            pool_cfg.add_input("conv_out");
            pool_cfg.add_output("pool_out");
            net_cfg.add_layer(pool_cfg);
            // set up linear1 layer
            let linear1_cfg = LinearConfig { output_size: 500 };
            let mut lnr1_cfg = LayerConfig::new("linear1", LayerType::Linear(linear1_cfg));
            lnr1_cfg.add_input("pool_out");
            lnr1_cfg.add_output("linear1_out");
            net_cfg.add_layer(lnr1_cfg);
            // set up relu layer
            let mut relu_cfg = LayerConfig::new("relu", LayerType::Sigmoid);
            relu_cfg.add_input("linear1_out");
            relu_cfg.add_output("relu_out");
            net_cfg.add_layer(relu_cfg);
            // set up linear2 layer
            let linear2_cfg = LinearConfig { output_size: 10 };
            let mut lnr2_cfg = LayerConfig::new("linear2", LayerType::Linear(linear2_cfg));
            lnr2_cfg.add_input("relu_out");
            lnr2_cfg.add_output("final_out");
            net_cfg.add_layer(lnr2_cfg);
        },
        "mlp" => {
            // set up reshape layer
            let mut reshape_cfg = LayerConfig::new("reshape", LayerType::Reshape(ReshapeConfig::of_shape(&vec![batch_size, 784])));
            reshape_cfg.add_input("data"); // provide the layer we want to reshape as bottom
            net_cfg.add_layer(reshape_cfg);
            // // set up linear1 layer
            let linear1_cfg = LinearConfig { output_size: 1568 };
            let mut lnr1_cfg = LayerConfig::new("linear1", LayerType::Linear(linear1_cfg));
            lnr1_cfg.add_input("data");
            lnr1_cfg.add_output("linear1_out");
            net_cfg.add_layer(lnr1_cfg);
            // set up sigmoid layer
            let mut sigmoid_cfg = LayerConfig::new("sigmoid", LayerType::Sigmoid);
            sigmoid_cfg.add_input("linear1_out");
            sigmoid_cfg.add_output("sigmoid_out");
            net_cfg.add_layer(sigmoid_cfg);
            // set up linear2 layer
            let linear2_cfg = LinearConfig { output_size: 10 };
            let mut lnr2_cfg = LayerConfig::new("linear2", LayerType::Linear(linear2_cfg));
            lnr2_cfg.add_input("sigmoid_out");
            lnr2_cfg.add_output("final_out");
            net_cfg.add_layer(lnr2_cfg);
        },
        "linear" => {
            // set up reshape layer
            let mut reshape_cfg = LayerConfig::new("reshape", LayerType::Reshape(ReshapeConfig::of_shape(&vec![batch_size, 784])));
            reshape_cfg.add_input("data"); // provide the layer we want to reshape as bottom
            net_cfg.add_layer(reshape_cfg);
            // set up linear layer
            let linear_cfg = LinearConfig { output_size: 10 };
            let mut lnr_cfg = LayerConfig::new("linear", LayerType::Linear(linear_cfg));
            lnr_cfg.add_input("data");
            lnr_cfg.add_output("final_out");
            net_cfg.add_layer(lnr_cfg);
        }
        _ => { panic!("Unknown model. Try one of [linear, mlp, conv]")}
    }
    // set up log_softmax
    let mut softmax_cfg = LayerConfig::new("log_softmax", LayerType::LogSoftmax);
    softmax_cfg.add_input("final_out");
    softmax_cfg.add_output("softmax_out");
    net_cfg.add_layer(softmax_cfg);
    // set up nll loss
    let nll_layer_cfg = NegativeLogLikelihoodConfig { num_classes: 10 };
    let mut nll_cfg = LayerConfig::new("nll", LayerType::NegativeLogLikelihood(nll_layer_cfg));
    nll_cfg.add_input("softmax_out");
    nll_cfg.add_input("label");
    net_cfg.add_layer(nll_cfg);

    let backend = ::std::rc::Rc::new(Backend::<Cuda>::default().unwrap());
    let native_backend = ::std::rc::Rc::new(Backend::<Native>::default().unwrap());
    let mut network = Network::from_config(backend.clone(), &net_cfg);
    // let mut network = Network::from_config(native_backend.clone(), &net_cfg);

    let mut inp = SharedTensor::<f32>::new(backend.device(), &vec![batch_size, 1, 28, 28]).unwrap();
    let label = SharedTensor::<f32>::new(native_backend.device(), &vec![batch_size]).unwrap();
    inp.add_device(native_backend.device()).unwrap();

    let inp_lock = Arc::new(RwLock::new(inp));
    let label_lock = Arc::new(RwLock::new(label));

    let mut prediction_score = Vec::new();
    for _ in 0..(60000 / batch_size) {
        // set up history; belongs in momentum
        let mut history = vec![];
        for weight_gradient in network.learnable_weight_gradients() {
            let shape = weight_gradient.read().unwrap().desc().clone();
            let history_tensor = Arc::new(RwLock::new(SharedTensor::<f32>::new(native_backend.device(), &shape).unwrap()));
            history.push(history_tensor);
        }
        // set up lr
        let mut lr_shared = SharedTensor::<f32>::new(native_backend.device(), &1).unwrap();
        write_to_memory_f32(lr_shared.get_mut(native_backend.device()).unwrap(), &vec![learning_rate], 0);
        // set up momentum
        let momentum = 0.9f32;
        let mut momentum_shared = SharedTensor::<f32>::new(native_backend.device(), &1).unwrap();
        write_to_memory_f32(momentum_shared.get_mut(native_backend.device()).unwrap(), &vec![momentum], 0);
        // clear weight gradients
        network.clear_weight_diffs();
        // write input
        let mut targets = Vec::new();
        for (batch_n, (label_val, input)) in decoded_images.by_ref().take(batch_size).enumerate() {
            {
                let inp_lock_cl = inp_lock.clone();
                let mut inp = inp_lock_cl.write().unwrap();
                inp.sync(native_backend.device()).unwrap();
                write_to_memory(inp.get_mut(native_backend.device()).unwrap(), &input, batch_n * 784);

                let label_lock_cl = label_lock.clone();
                let mut label = label_lock_cl.write().unwrap();
                label.sync(native_backend.device()).unwrap();
                write_to_memory(label.get_mut(native_backend.device()).unwrap(), &[label_val], batch_n);

                targets.push(label_val);
            }
        }
        let _ = network.forward_backward(&[inp_lock.clone(), label_lock.clone()]);

        let softmax_out = network.get_data("softmax_out");
        let sftmax_out = softmax_out.read().unwrap();
        let native_softmax_out = sftmax_out.get(native_backend.device()).unwrap().as_native().unwrap();
        // let mut enumerated_predictions = native_softmax_out.as_slice::<f32>().iter().enumerate().collect::<Vec<_>>();
        let predictions_slice = native_softmax_out.as_slice::<f32>();
        let mut predictions = Vec::new();
        for batch_predictions in predictions_slice.chunks(10) {
            let mut enumerated_predictions = batch_predictions.iter().enumerate().collect::<Vec<_>>();
            enumerated_predictions.sort_by(|&(_, one), &(_, two)| one.partial_cmp(two).unwrap_or(::std::cmp::Ordering::Equal)); // find index of prediction
            predictions.push(enumerated_predictions.last().unwrap().0)
            // let prediction = batch_predictions.last().unwrap().0;
        }

        for ((weight_gradient, history_tensor), weight_data) in network.learnable_weight_gradients().iter().zip(history).zip(network.learnable_weight_data()) {
            weight_gradient.write().unwrap().sync(native_backend.device()).unwrap();
            ::leaf::util::Axpby::<f32>::axpby_plain(&*native_backend,
                                                   &lr_shared,
                                                   &weight_gradient.read().unwrap(),
                                                   &momentum_shared,
                                                   &mut history_tensor.write().unwrap()).unwrap();
            native_backend.copy_plain(
                &history_tensor.read().unwrap(), &mut weight_gradient.write().unwrap()).unwrap();

            let _ = weight_gradient.write().unwrap().add_device(native_backend.device());
            weight_gradient.write().unwrap().sync(native_backend.device()).unwrap();
            let _ = history_tensor.write().unwrap().add_device(native_backend.device());
            history_tensor.write().unwrap().sync(native_backend.device()).unwrap();
            let _ = weight_data.write().unwrap().add_device(native_backend.device());
            weight_data.write().unwrap().sync(native_backend.device()).unwrap();
        }
        // network.update_weights(&*backend);
        network.update_weights(&*native_backend);

        for (&prediction, &target) in predictions.iter().zip(targets.iter()) {
            prediction_score.push(prediction == target as usize);
            let correct_predictions = prediction_score.iter().filter(|&&v| v == true).count();
            let accuracy = (correct_predictions as f32) / (prediction_score.len() as f32) * 100f32;
            println!("Prediction: {:?}, Target: {:?} | Accuracy {:?} / {:?} , {:.2?}%", prediction, target, correct_predictions, prediction_score.len(), accuracy);
        }
    }
}

fn write_to_memory(mem: &mut MemoryType, data: &[u8], offset: usize) {
    if let &mut MemoryType::Native(ref mut mem) = mem {
        let mut mem_buffer = mem.as_mut_slice::<f32>();
        for (index, datum) in data.iter().enumerate() {
            mem_buffer[index + offset] = *datum as f32;
        }
    }
}

fn write_to_memory_f32(mem: &mut MemoryType, data: &[f32], offset: usize) {
    if let &mut MemoryType::Native(ref mut mem) = mem {
        let mut mem_buffer = mem.as_mut_slice::<f32>();
        for (index, datum) in data.iter().enumerate() {
            mem_buffer[index + offset] = *datum as f32;
        }
    }
}
