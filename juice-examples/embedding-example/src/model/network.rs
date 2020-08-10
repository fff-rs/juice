use coaster::frameworks::cuda::get_cuda_backend;
use coaster::prelude::*;
use coaster_nn::{DirectionMode, RnnInputMode, RnnNetworkMode};
use juice::layer::*;
use juice::layers::*;
use juice::solver::*;
use juice::util::*;
use crate::model::params::{EMBED_DIMENSION, VOCAB_SIZE, PHRASE_LENGTH};

pub(crate) fn create_network(batch_size: usize, sequence_size: usize, outputs: usize) -> SequentialConfig {
    let mut net_cfg = SequentialConfig::default();
    net_cfg.add_input("data_input", &[batch_size, sequence_size]);
    net_cfg.force_backward = true;

    net_cfg.add_layer(LayerConfig::new(
        "embedding",
        LayerType::Embedding(EmbeddingConfig {
            embedding_dimension: EMBED_DIMENSION,
            vocab_size: VOCAB_SIZE,
            phrase_length: sequence_size
        }),
    ));

    net_cfg.add_layer(LayerConfig::new(
        "reshape",
        ReshapeConfig::of_shape(&[batch_size, 1, EMBED_DIMENSION, PHRASE_LENGTH]),
    ));

    net_cfg.add_layer(LayerConfig::new(
        "pooling",
        PoolingConfig {
            mode: PoolingMode::Max,
            filter_shape: vec![3],
            padding: vec![1],
            stride: vec![2],
        },
    ));

    net_cfg.add_layer(LayerConfig::new(
        "conv1",
        ConvolutionConfig {
            num_output: 50,
            filter_shape: vec![3],
            padding: vec![1],
            stride: vec![2],
        },
    ));

    net_cfg.add_layer(LayerConfig::new(
        "pooling",
        PoolingConfig {
            mode: PoolingMode::Max,
            filter_shape: vec![3],
            padding: vec![1],
            stride: vec![2],
        },
    ));

    net_cfg.add_layer(LayerConfig::new("linear1", LinearConfig { output_size: outputs }));
    net_cfg.add_layer(LayerConfig::new("softmax", LayerType::Softmax));
    net_cfg
}