#![cfg(feature = "cuda")]

use coaster::frameworks::cuda::get_cuda_backend as cuda_backend;
use criterion::{criterion_group, criterion_main, Criterion};

use juice::{net::*, weight::FillerType};

pub fn bench_mnsit_forward_1(c: &mut Criterion) {
    c.bench_function("bench_mnsit_forward_1", |b| {
        let backend = cuda_backend();

        let mut net_cfg = SequentialConfig::default();
        net_cfg.add_layer("sigmoid", LayerConfig::Sigmoid);
        net_cfg.add_layer("linear", LinearConfig { output_size: 10 });

        let net = Network::from_config(&backend, net_cfg, &[vec![30, 30]]).unwrap();

        let mut context = Context::new(1);
        let input = context.acquire_data(net.top().descriptor().input(0));
        FillerType::fill_constant(&mut input.borrow_mut(), 0.0);

        b.iter(|| {
            net.top().compute_output(&backend, &mut context);
        })
    });
}

pub fn alexnet_forward(c: &mut Criterion) {
    c.bench_function("alexnet_forward", |b| {
        let backend = cuda_backend();

        let mut net_cfg = SequentialConfig::default();

        net_cfg.add_layer(
            "conv1",
            ConvolutionConfig {
                feature_maps: 64,
                kernel_size: 11,
                padding: 2,
                stride: 4,
            },
        );
        net_cfg.add_layer("relu1", LayerConfig::Relu);
        net_cfg.add_layer(
            "pool1",
            PoolingConfig {
                mode: PoolingMode::Max,
                window_size: 3,
                padding: 0,
                stride: 2,
            },
        );

        net_cfg.add_layer(
            "conv2",
            ConvolutionConfig {
                feature_maps: 192,
                kernel_size: 5,
                padding: 2,
                stride: 1,
            },
        );
        net_cfg.add_layer("relu2", LayerConfig::Relu);
        net_cfg.add_layer(
            "pool2",
            PoolingConfig {
                mode: PoolingMode::Max,
                window_size: 3,
                padding: 0,
                stride: 2,
            },
        );

        net_cfg.add_layer(
            "conv3",
            ConvolutionConfig {
                feature_maps: 384,
                kernel_size: 3,
                padding: 1,
                stride: 1,
            },
        );
        net_cfg.add_layer("relu3", LayerConfig::Relu);

        net_cfg.add_layer(
            "conv4",
            ConvolutionConfig {
                feature_maps: 256,
                kernel_size: 3,
                padding: 1,
                stride: 1,
            },
        );
        net_cfg.add_layer("relu4", LayerConfig::Relu);

        net_cfg.add_layer(
            "conv5",
            ConvolutionConfig {
                feature_maps: 256,
                kernel_size: 3,
                padding: 1,
                stride: 1,
            },
        );
        net_cfg.add_layer("relu5", LayerConfig::Relu);

        net_cfg.add_layer(
            "pool3",
            PoolingConfig {
                mode: PoolingMode::Max,
                window_size: 3,
                padding: 0,
                stride: 2,
            },
        );

        net_cfg.add_layer("fc1", LinearConfig { output_size: 4096 });
        net_cfg.add_layer("fc2", LinearConfig { output_size: 4096 });
        net_cfg.add_layer("fc3", LinearConfig { output_size: 1000 });

        let net = Network::from_config(&backend, net_cfg, &[vec![3, 224, 224]]).unwrap();

        let mut context = Context::new(128);
        let input = context.acquire_data(net.top().descriptor().input(0));
        FillerType::fill_constant(&mut input.borrow_mut(), 0.0);

        b.iter(|| {
            net.top().compute_output(&backend, &mut context);
        })
    });
}

pub fn small_alexnet_forward(c: &mut Criterion) {
    c.bench_function("small_alexnet_forward", |b| {
        let backend = cuda_backend();

        let mut net_cfg = SequentialConfig::default();

        net_cfg.add_layer(
            "conv1",
            ConvolutionConfig {
                feature_maps: 32,
                kernel_size: 11,
                padding: 2,
                stride: 4,
            },
        );
        net_cfg.add_layer("relu1", LayerConfig::Relu);
        net_cfg.add_layer(
            "pool1",
            PoolingConfig {
                mode: PoolingMode::Max,
                window_size: 3,
                padding: 0,
                stride: 2,
            },
        );
        
        net_cfg.add_layer(
            "conv2",
            ConvolutionConfig {
                feature_maps: 96,
                kernel_size: 5,
                padding: 2,
                stride: 1,
            },
        );
        net_cfg.add_layer("relu2", LayerConfig::Relu);
        net_cfg.add_layer(
            "pool2",
            PoolingConfig {
                mode: PoolingMode::Max,
                window_size: 3,
                padding: 0,
                stride: 2,
            },
        );
        
        net_cfg.add_layer(
            "conv3",
            ConvolutionConfig {
                feature_maps: 142,
                kernel_size: 3,
                padding: 1,
                stride: 1,
            },
        );
        net_cfg.add_layer("relu3", LayerConfig::Relu);

        net_cfg.add_layer(
            "conv4",
            ConvolutionConfig {
                feature_maps: 128,
                kernel_size: 3,
                padding: 1,
                stride: 1,
            },
        );
        net_cfg.add_layer("relu4", LayerConfig::Relu);

        net_cfg.add_layer(
            "conv5",
            ConvolutionConfig {
                feature_maps: 128,
                kernel_size: 3,
                padding: 1,
                stride: 1,
            },
        );
        net_cfg.add_layer("relu5", LayerConfig::Relu);

        net_cfg.add_layer(
            "pool3",
            PoolingConfig {
                mode: PoolingMode::Max,
                window_size: 3,
                padding: 0,
                stride: 2,
            },
        );

        net_cfg.add_layer("fc1", LinearConfig { output_size: 2048 });
        net_cfg.add_layer("fc2", LinearConfig { output_size: 2048 });
        net_cfg.add_layer("fc3", LinearConfig { output_size: 500 });

        let net = Network::from_config(&backend, net_cfg, &[vec![3, 112, 112]]).unwrap();

        let mut context = Context::new(128);
        let input = context.acquire_data(net.top().descriptor().input(0));
        FillerType::fill_constant(&mut input.borrow_mut(), 0.0);

        b.iter(|| {
            net.top().compute_output(&backend, &mut context);
        })
    });

}
 
criterion_group!(benches, bench_mnsit_forward_1, alexnet_forward, small_alexnet_forward);
criterion_main!(benches);
