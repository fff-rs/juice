#![feature(test)]
#![feature(clone_from_slice)]

extern crate test;
#[macro_use]
extern crate timeit;
extern crate collenchyma as co;
extern crate leaf;

use test::Bencher;
use std::io::Write;
use co::device::{IDevice, DeviceType};
use co::backend::{Backend, BackendConfig};
#[cfg(feature = "native")]
use co::frameworks::Native;
#[cfg(feature = "native")]
use co::frameworks::native;
#[cfg(feature = "opencl")]
use co::frameworks::OpenCL;
#[cfg(feature = "opencl")]
use co::frameworks::opencl;
#[cfg(feature = "cuda")]
use co::frameworks::Cuda;
#[cfg(feature = "cuda")]
use co::frameworks::cuda;
use co::framework::IFramework;
use co::tensor::SharedTensor;

use std::sync::{Arc, RwLock};
use leaf::shared_memory::*;
use leaf::layers::*;
use leaf::layer::*;
use leaf::network::*;
use std::rc::Rc;

#[cfg(feature = "native")]
fn native_backend() -> Rc<Backend<Native>> {
    let framework = Native::new();
    let hardwares = &framework.hardwares().to_vec();
    let backend_config = BackendConfig::new(framework, hardwares);
    Rc::new(Backend::new(backend_config).unwrap())
}

#[cfg(feature = "cuda")]
fn cuda_backend() -> Rc<Backend<Cuda>> {
    let framework = Cuda::new();
    let hardwares = &framework.hardwares()[0..1].to_vec();
    let backend_config = BackendConfig::new(framework, hardwares);
    Rc::new(Backend::new(backend_config).unwrap())
}

#[cfg(feature = "opencl")]
fn opencl_backend() -> Rc<Backend<OpenCL>> {
    let framework = OpenCL::new();
    let hardwares = &framework.hardwares()[1..2].to_vec();
    let backend_config = BackendConfig::new(framework, hardwares);
    Rc::new(Backend::new(backend_config).unwrap())
}

#[inline(never)]
fn bench_profile<F: FnMut() -> ()>(
    b: &mut Bencher,
    mut bench_func: F,
    times: usize) {
    println!("toastyhere");
    writeln!(&mut ::std::io::stderr(), "here"); // write to stderr
    // let sec = timeit_loops!(times, {
    //     println!("there");
    //     bench_func();
    // });
    // b.bench_n(times as u64, || { bench_func(); });
}

// #[inline(never)]
// fn sync_back_and_forth(
//     b: &mut Bencher,
//     n: usize,
//     nt_device: &DeviceType,
//     cl_device: &DeviceType,
//     mem: &mut SharedTensor<u8>
// ) {
//     b.iter(|| {
//         for _ in 0..n {
//             match mem.sync(&cl_device) {
//                 Ok(_) => assert!(true),
//                 Err(err) => {
//                     println!("{:?}", err);
//                     assert!(false);
//                 }
//             }
//             match mem.sync(&nt_device) {
//                 Ok(_) => assert!(true),
//                 Err(err) => {
//                     println!("{:?}", err);
//                     assert!(false);
//                 }
//             }
//         }
//     });
// }

#[bench]
#[ignore]
#[cfg(feature = "cuda")]
fn bench_mnsit_forward_1(b: &mut Bencher) {
    let mut cfg = NetworkConfig::default();
    // set up input
    cfg.inputs.push("in".to_owned());
    cfg.input_shapes.push(vec![1, 30, 30]);
    cfg.inputs.push("label".to_owned());
    cfg.input_shapes.push(vec![1, 1, 10]);
    // set up sigmoid
    let mut sig_cfg = LayerConfig::new("sig".to_owned(), LayerType::Sigmoid);
    sig_cfg.bottoms.push("in".to_owned());
    sig_cfg.tops.push("sig_out".to_owned());
    cfg.layers.push(sig_cfg);

    let mut fc_layer_cfg = FullyConnectedConfig {
        num_output: 10,
        axis: None,
    };
    let mut fc_cfg = LayerConfig::new("fully_connected".to_owned(), LayerType::FullyConnected(fc_layer_cfg));
    fc_cfg.bottoms.push("sig_out".to_owned());
    fc_cfg.tops.push("fc_out".to_owned());
    cfg.layers.push(fc_cfg);
    // set up softmax_loss
    let mut loss_cfg = LayerConfig::new("loss".to_owned(), LayerType::SoftmaxLoss);
    loss_cfg.bottoms.push("fc_out".to_owned());
    loss_cfg.bottoms.push("label".to_owned());
    cfg.layers.push(loss_cfg);

    let backend = cuda_backend();
    let native_backend = native_backend();
    let mut network = Network::from_config(backend.clone(), &cfg);
    let loss = &mut 0f32;

    let time = timeit_loops!(10, {
        let inp = Blob::from_data(SharedTensor::<f32>::new(backend.device(), &vec![1, 30, 30]).unwrap());
        let label = Blob::from_data(SharedTensor::<f32>::new(native_backend.device(), &vec![1, 1, 10]).unwrap());

        let inp_lock = Arc::new(RwLock::new(inp));
        let label_lock = Arc::new(RwLock::new(label));

        network.forward(&[inp_lock, label_lock], loss);
    });
    // b.iter(|| {
    //     for _ in 0..1 {
    //         let inp = Blob::from_data(SharedTensor::<f32>::new(backend.device(), &vec![1, 30, 30]).unwrap());
    //         let label = Blob::from_data(SharedTensor::<f32>::new(native_backend.device(), &vec![1, 1, 10]).unwrap());
    //
    //         let inp_lock = Arc::new(RwLock::new(inp));
    //         let label_lock = Arc::new(RwLock::new(label));
    //
    //         network.forward(&[inp_lock, label_lock], loss);
    //     }
    // });
}

#[bench]
// #[ignore]
#[cfg(feature = "cuda")]
fn alexnet_forward(b: &mut Bencher) {
    // let _ = env_logger::init();
    let mut cfg = NetworkConfig::default();
    // Layer: data
    cfg.inputs.push("data".to_owned());
    cfg.input_shapes.push(vec![128, 3, 224, 224]);
    // Layer: conv1
    let conv1_layer_cfg = ConvolutionConfig {
        num_output: 64,
        filter_shape: vec![11],
        padding: vec![2],
        stride: vec![4],
        axis: None
    };
    let mut conv1_cfg = LayerConfig::new("conv1".to_owned(), LayerType::Convolution(conv1_layer_cfg));
    conv1_cfg.bottoms.push("data".to_owned());
    conv1_cfg.tops.push("conv1_preac".to_owned());
    cfg.layers.push(conv1_cfg);
    // Layer: conv1/relu
    let mut conv1_relu_cfg = LayerConfig::new("conv1/relu".to_owned(), LayerType::ReLU);
    conv1_relu_cfg.bottoms.push("conv1_preac".to_owned());
    conv1_relu_cfg.tops.push("conv1_out".to_owned());
    cfg.layers.push(conv1_relu_cfg);
    // Layer: pool1
    let pool1_layer_cfg = PoolingConfig {
        mode: PoolingMode::Max,
        filter_shape: vec![3],
        stride: vec![2],
        padding: vec![0], // TODO: make optional
        axis: None
    };
    let mut pool1_cfg = LayerConfig::new("pool1".to_owned(), LayerType::Pooling(pool1_layer_cfg));
    pool1_cfg.bottoms.push("conv1_out".to_owned());
    pool1_cfg.tops.push("pool1_out".to_owned());
    cfg.layers.push(pool1_cfg);
    // Layer: conv2
    let conv2_layer_cfg = ConvolutionConfig {
        num_output: 192,
        filter_shape: vec![5],
        padding: vec![2],
        stride: vec![1],
        axis: None
    };
    let mut conv2_cfg = LayerConfig::new("conv2".to_owned(), LayerType::Convolution(conv2_layer_cfg));
    conv2_cfg.bottoms.push("pool1_out".to_owned());
    conv2_cfg.tops.push("conv2_preac".to_owned());
    cfg.layers.push(conv2_cfg);
    // Layer: conv2/relu
    let mut conv2_relu_cfg = LayerConfig::new("conv2/relu".to_owned(), LayerType::ReLU);
    conv2_relu_cfg.bottoms.push("conv2_preac".to_owned());
    conv2_relu_cfg.tops.push("conv2_out".to_owned());
    cfg.layers.push(conv2_relu_cfg);
    // Layer: pool2
    let pool2_layer_cfg = PoolingConfig {
        mode: PoolingMode::Max,
        filter_shape: vec![3],
        stride: vec![2],
        padding: vec![0], // TODO: make optional
        axis: None
    };
    let mut pool2_cfg = LayerConfig::new("pool2".to_owned(), LayerType::Pooling(pool2_layer_cfg));
    pool2_cfg.bottoms.push("conv2_out".to_owned());
    pool2_cfg.tops.push("pool2_out".to_owned());
    cfg.layers.push(pool2_cfg);
    // Layer: conv3
    let conv3_layer_cfg = ConvolutionConfig {
        num_output: 384,
        filter_shape: vec![3],
        padding: vec![1],
        stride: vec![1],
        axis: None
    };
    let mut conv3_cfg = LayerConfig::new("conv3".to_owned(), LayerType::Convolution(conv3_layer_cfg));
    conv3_cfg.bottoms.push("pool2_out".to_owned());
    conv3_cfg.tops.push("conv3_preac".to_owned());
    cfg.layers.push(conv3_cfg);
    // Layer: conv3/relu
    let mut conv3_relu_cfg = LayerConfig::new("conv3/relu".to_owned(), LayerType::ReLU);
    conv3_relu_cfg.bottoms.push("conv3_preac".to_owned());
    conv3_relu_cfg.tops.push("conv3_out".to_owned());
    cfg.layers.push(conv3_relu_cfg);
    // Layer: conv4
    let conv4_layer_cfg = ConvolutionConfig {
        num_output: 256,
        filter_shape: vec![3],
        padding: vec![1],
        stride: vec![1],
        axis: None
    };
    let mut conv4_cfg = LayerConfig::new("conv4".to_owned(), LayerType::Convolution(conv4_layer_cfg));
    conv4_cfg.bottoms.push("conv3_out".to_owned());
    conv4_cfg.tops.push("conv4_preac".to_owned());
    cfg.layers.push(conv4_cfg);
    // Layer: conv4/relu
    let mut conv4_relu_cfg = LayerConfig::new("conv4/relu".to_owned(), LayerType::ReLU);
    conv4_relu_cfg.bottoms.push("conv4_preac".to_owned());
    conv4_relu_cfg.tops.push("conv4_out".to_owned());
    cfg.layers.push(conv4_relu_cfg);
    // Layer: conv5
    let conv5_layer_cfg = ConvolutionConfig {
        num_output: 256,
        filter_shape: vec![3],
        padding: vec![1],
        stride: vec![1],
        axis: None
    };
    let mut conv5_cfg = LayerConfig::new("conv5".to_owned(), LayerType::Convolution(conv5_layer_cfg));
    conv5_cfg.bottoms.push("conv4_out".to_owned());
    conv5_cfg.tops.push("conv5_preac".to_owned());
    cfg.layers.push(conv5_cfg);
    // Layer: conv5/relu
    let mut conv5_relu_cfg = LayerConfig::new("conv5/relu".to_owned(), LayerType::ReLU);
    conv5_relu_cfg.bottoms.push("conv5_preac".to_owned());
    conv5_relu_cfg.tops.push("conv5_out".to_owned());
    cfg.layers.push(conv5_relu_cfg);
    // Layer: pool3
    let pool3_layer_cfg = PoolingConfig {
        mode: PoolingMode::Max,
        filter_shape: vec![3],
        stride: vec![2],
        padding: vec![0], // TODO: make optional
        axis: None
    };
    let mut pool3_cfg = LayerConfig::new("pool3".to_owned(), LayerType::Pooling(pool3_layer_cfg));
    pool3_cfg.bottoms.push("conv5_out".to_owned());
    pool3_cfg.tops.push("pool3_out".to_owned());
    cfg.layers.push(pool3_cfg);
    // Layer: fc1
    let mut fc1_layer_cfg = FullyConnectedConfig {
        num_output: 4096,
        axis: None,
    };
    let mut fc1_cfg = LayerConfig::new("fc1".to_owned(), LayerType::FullyConnected(fc1_layer_cfg));
    fc1_cfg.bottoms.push("pool3_out".to_owned());
    fc1_cfg.tops.push("fc1_out".to_owned());
    cfg.layers.push(fc1_cfg);
    // Layer: fc2
    let mut fc2_layer_cfg = FullyConnectedConfig {
        num_output: 4096,
        axis: None,
    };
    let mut fc2_cfg = LayerConfig::new("fc2".to_owned(), LayerType::FullyConnected(fc2_layer_cfg));
    fc2_cfg.bottoms.push("fc1_out".to_owned());
    fc2_cfg.tops.push("fc2_out".to_owned());
    cfg.layers.push(fc2_cfg);
    // Layer: fc3
    let mut fc3_layer_cfg = FullyConnectedConfig {
        num_output: 1000,
        axis: None,
    };
    let mut fc3_cfg = LayerConfig::new("fc3".to_owned(), LayerType::FullyConnected(fc3_layer_cfg));
    fc3_cfg.bottoms.push("fc2_out".to_owned());
    fc3_cfg.tops.push("fc3_out".to_owned());
    cfg.layers.push(fc3_cfg);

    let backend = cuda_backend();
    let native_backend = native_backend();
    let mut network = Network::from_config(backend.clone(), &cfg);

    let mut func = || {
        let forward_time = timeit_loops!(1, {
            let loss = &mut 0f32;
            let inp = Blob::from_data(SharedTensor::<f32>::new(backend.device(), &vec![128, 3, 112, 112]).unwrap());

            let inp_lock = Arc::new(RwLock::new(inp));
            network.forward(&[inp_lock], loss);
        });
        println!("Forward step: {}", forward_time);
    };
    { bench_profile(b, func, 10); }
}

#[bench]
#[ignore]
#[cfg(feature = "cuda")]
fn small_alexnet_forward(b: &mut Bencher) {
    // let _ = env_logger::init();
    let mut cfg = NetworkConfig::default();
    // Layer: data
    cfg.inputs.push("data".to_owned());
    cfg.input_shapes.push(vec![128, 3, 112, 112]);
    // Layer: conv1
    let conv1_layer_cfg = ConvolutionConfig {
        num_output: 32,
        filter_shape: vec![11],
        padding: vec![2],
        stride: vec![4],
        axis: None
    };
    let mut conv1_cfg = LayerConfig::new("conv1".to_owned(), LayerType::Convolution(conv1_layer_cfg));
    conv1_cfg.bottoms.push("data".to_owned());
    conv1_cfg.tops.push("conv1_preac".to_owned());
    cfg.layers.push(conv1_cfg);
    // Layer: conv1/relu
    let mut conv1_relu_cfg = LayerConfig::new("conv1/relu".to_owned(), LayerType::ReLU);
    conv1_relu_cfg.bottoms.push("conv1_preac".to_owned());
    conv1_relu_cfg.tops.push("conv1_out".to_owned());
    cfg.layers.push(conv1_relu_cfg);
    // Layer: pool1
    let pool1_layer_cfg = PoolingConfig {
        mode: PoolingMode::Max,
        filter_shape: vec![3],
        stride: vec![2],
        padding: vec![0], // TODO: make optional
        axis: None
    };
    let mut pool1_cfg = LayerConfig::new("pool1".to_owned(), LayerType::Pooling(pool1_layer_cfg));
    pool1_cfg.bottoms.push("conv1_out".to_owned());
    pool1_cfg.tops.push("pool1_out".to_owned());
    cfg.layers.push(pool1_cfg);
    // Layer: conv2
    let conv2_layer_cfg = ConvolutionConfig {
        num_output: 96,
        filter_shape: vec![5],
        padding: vec![2],
        stride: vec![1],
        axis: None
    };
    let mut conv2_cfg = LayerConfig::new("conv2".to_owned(), LayerType::Convolution(conv2_layer_cfg));
    conv2_cfg.bottoms.push("pool1_out".to_owned());
    conv2_cfg.tops.push("conv2_preac".to_owned());
    cfg.layers.push(conv2_cfg);
    // Layer: conv2/relu
    let mut conv2_relu_cfg = LayerConfig::new("conv2/relu".to_owned(), LayerType::ReLU);
    conv2_relu_cfg.bottoms.push("conv2_preac".to_owned());
    conv2_relu_cfg.tops.push("conv2_out".to_owned());
    cfg.layers.push(conv2_relu_cfg);
    // Layer: pool2
    let pool2_layer_cfg = PoolingConfig {
        mode: PoolingMode::Max,
        filter_shape: vec![3],
        stride: vec![2],
        padding: vec![0], // TODO: make optional
        axis: None
    };
    let mut pool2_cfg = LayerConfig::new("pool2".to_owned(), LayerType::Pooling(pool2_layer_cfg));
    pool2_cfg.bottoms.push("conv2_out".to_owned());
    pool2_cfg.tops.push("pool2_out".to_owned());
    cfg.layers.push(pool2_cfg);
    // Layer: conv3
    let conv3_layer_cfg = ConvolutionConfig {
        num_output: 142,
        filter_shape: vec![3],
        padding: vec![1],
        stride: vec![1],
        axis: None
    };
    let mut conv3_cfg = LayerConfig::new("conv3".to_owned(), LayerType::Convolution(conv3_layer_cfg));
    conv3_cfg.bottoms.push("pool2_out".to_owned());
    conv3_cfg.tops.push("conv3_preac".to_owned());
    cfg.layers.push(conv3_cfg);
    // Layer: conv3/relu
    let mut conv3_relu_cfg = LayerConfig::new("conv3/relu".to_owned(), LayerType::ReLU);
    conv3_relu_cfg.bottoms.push("conv3_preac".to_owned());
    conv3_relu_cfg.tops.push("conv3_out".to_owned());
    cfg.layers.push(conv3_relu_cfg);
    // Layer: conv4
    let conv4_layer_cfg = ConvolutionConfig {
        num_output: 128,
        filter_shape: vec![3],
        padding: vec![1],
        stride: vec![1],
        axis: None
    };
    let mut conv4_cfg = LayerConfig::new("conv4".to_owned(), LayerType::Convolution(conv4_layer_cfg));
    conv4_cfg.bottoms.push("conv3_out".to_owned());
    conv4_cfg.tops.push("conv4_preac".to_owned());
    cfg.layers.push(conv4_cfg);
    // Layer: conv4/relu
    let mut conv4_relu_cfg = LayerConfig::new("conv4/relu".to_owned(), LayerType::ReLU);
    conv4_relu_cfg.bottoms.push("conv4_preac".to_owned());
    conv4_relu_cfg.tops.push("conv4_out".to_owned());
    cfg.layers.push(conv4_relu_cfg);
    // Layer: conv5
    let conv5_layer_cfg = ConvolutionConfig {
        num_output: 128,
        filter_shape: vec![3],
        padding: vec![1],
        stride: vec![1],
        axis: None
    };
    let mut conv5_cfg = LayerConfig::new("conv5".to_owned(), LayerType::Convolution(conv5_layer_cfg));
    conv5_cfg.bottoms.push("conv4_out".to_owned());
    conv5_cfg.tops.push("conv5_preac".to_owned());
    cfg.layers.push(conv5_cfg);
    // Layer: conv5/relu
    let mut conv5_relu_cfg = LayerConfig::new("conv5/relu".to_owned(), LayerType::ReLU);
    conv5_relu_cfg.bottoms.push("conv5_preac".to_owned());
    conv5_relu_cfg.tops.push("conv5_out".to_owned());
    cfg.layers.push(conv5_relu_cfg);
    // Layer: pool3
    let pool3_layer_cfg = PoolingConfig {
        mode: PoolingMode::Max,
        filter_shape: vec![3],
        stride: vec![2],
        padding: vec![0], // TODO: make optional
        axis: None
    };
    let mut pool3_cfg = LayerConfig::new("pool3".to_owned(), LayerType::Pooling(pool3_layer_cfg));
    pool3_cfg.bottoms.push("conv5_out".to_owned());
    pool3_cfg.tops.push("pool3_out".to_owned());
    cfg.layers.push(pool3_cfg);
    // Layer: fc1
    let mut fc1_layer_cfg = FullyConnectedConfig {
        num_output: 2048,
        axis: None,
    };
    let mut fc1_cfg = LayerConfig::new("fc1".to_owned(), LayerType::FullyConnected(fc1_layer_cfg));
    fc1_cfg.bottoms.push("pool3_out".to_owned());
    fc1_cfg.tops.push("fc1_out".to_owned());
    cfg.layers.push(fc1_cfg);
    // Layer: fc2
    let mut fc2_layer_cfg = FullyConnectedConfig {
        num_output: 2048,
        axis: None,
    };
    let mut fc2_cfg = LayerConfig::new("fc2".to_owned(), LayerType::FullyConnected(fc2_layer_cfg));
    fc2_cfg.bottoms.push("fc1_out".to_owned());
    fc2_cfg.tops.push("fc2_out".to_owned());
    cfg.layers.push(fc2_cfg);
    // Layer: fc3
    let mut fc3_layer_cfg = FullyConnectedConfig {
        num_output: 500,
        axis: None,
    };
    let mut fc3_cfg = LayerConfig::new("fc3".to_owned(), LayerType::FullyConnected(fc3_layer_cfg));
    fc3_cfg.bottoms.push("fc2_out".to_owned());
    fc3_cfg.tops.push("fc3_out".to_owned());
    cfg.layers.push(fc3_cfg);

    let backend = cuda_backend();
    let native_backend = native_backend();
    let mut network = Network::from_config(backend.clone(), &cfg);

    let mut func = || {
        let loss = &mut 0f32;
        let inp = Blob::from_data(SharedTensor::<f32>::new(backend.device(), &vec![128, 3, 112, 112]).unwrap());

        let inp_lock = Arc::new(RwLock::new(inp));
        network.forward(&[inp_lock], loss);
    };
    { func(); bench_profile(b, func, 10); }
}
