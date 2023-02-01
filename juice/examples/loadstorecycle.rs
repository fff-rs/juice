use std::{io::BufReader, fs::File};

use juice::{layers::*, capnp_util};
use juice::layer::{LayerType, LayerConfig};
use juice::capnp_util::{CapnpRead, CapnpWrite};

fn main() {
    let mut cfg = SequentialConfig::default();
    cfg.add_input("data", &[64, 3, 224, 224]);

    let conv1_layer_cfg = ConvolutionConfig {
        num_output: 64,
        filter_shape: vec![3],
        padding: vec![1],
        stride: vec![1],
    };
    cfg.add_layer(LayerConfig::new("conv1", conv1_layer_cfg));
    cfg.add_layer(LayerConfig::new("conv1/relu", LayerType::ReLU));
    cfg.add_layer(LayerConfig::new(
        "pool1",
        PoolingConfig {
            mode: PoolingMode::Max,
            filter_shape: vec![2],
            stride: vec![2],
            padding: vec![0],
        },
    ));

    cfg.add_layer(LayerConfig::new(
        "conv2",
        ConvolutionConfig {
            num_output: 128,
            filter_shape: vec![3],
            padding: vec![1],
            stride: vec![1],
        },
    ));
    cfg.add_layer(LayerConfig::new("conv2/relu", LayerType::ReLU));
    cfg.add_layer(LayerConfig::new("fc1", LinearConfig { output_size: 2000 }));
    cfg.add_layer(LayerConfig::new("fc2", LinearConfig { output_size: 100 }));
    cfg.add_layer(LayerConfig::new("fc3", LinearConfig { output_size: 2 }));

    
    let p = "./foo.serialized.capnp";
    {
        let mut f = File::options().truncate(true).create(true).write(true).open(p).unwrap();
        // let mut builder = juice::juice_capnp::sequential_config::Builder;
        let mut builder = capnp::message::TypedBuilder::<juice::juice_capnp::sequential_config::Owned>::new_default();
        let facade = &mut builder.get_root().unwrap();
        cfg.write_capnp(facade);
    
        capnp::serialize::write_message(&mut f, builder.borrow_inner()).unwrap();
    }
    let reincarnation = {
        let f = File::options().read(true).open(p).unwrap();

        let reader = BufReader::new(f);
        let reader = capnp::serialize::try_read_message(
            reader,
            capnp::message::ReaderOptions {
                traversal_limit_in_words: None,
                nesting_limit: 100,
            }).unwrap().unwrap();
        <SequentialConfig as CapnpRead>::read_capnp(reader.get_root().unwrap())
    };
    
    assert_eq!(dbg!(cfg), dbg!(reincarnation));
    
}
