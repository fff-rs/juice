use std::{fs::File, io::BufReader};

use juice::util::native_backend;
use juice::{capnp_util, layer::*, layers::*, solver::*};
// use juice::layer::{LayerType, LayerConfig, Layer};
use coaster::prelude::*;
use juice::capnp_util::{CapnpRead, CapnpWrite};
use juice::util::{LayerOps, SolverOps};
use num::cast;
use num::NumCast;
use rand::distributions::Distribution;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::RwLock;

fn get_native_backend() -> Backend<Native> {
    Backend::<Native>::default().unwrap()
}

fn sillynet() -> SequentialConfig {
    let mut cfg = SequentialConfig::default();
    cfg.add_input("data", &[1, 2, 3, 2]);

    cfg.add_layer(LayerConfig::new("fc1", LinearConfig { output_size: 13 }));
    cfg.add_layer(LayerConfig::new("fc2", LinearConfig { output_size: 7 }));
    cfg.add_layer(LayerConfig::new("fc3", LinearConfig { output_size: 1 }));
    cfg
}

fn add_solver<Framework: IFramework + 'static>(
    backend: Rc<Backend<Framework>>,
    net_cfg: SequentialConfig,
    batch_size: usize,
    learning_rate: f32,
    momentum: f32,
) -> Solver<Backend<Framework>, Backend<Framework>>
where
    Backend<Framework>: coaster::IBackend + SolverOps<f32> + LayerOps<f32>,
{
    // Define an Objective Function
    let mut regressor_cfg = SequentialConfig::default();

    // Bit confusing, but the output is seen as the same as the input?
    regressor_cfg.add_input("data_output", &[batch_size, 1]);
    regressor_cfg.add_input("label", &[batch_size, 1]);

    // Add a Layer expressing Mean Squared Error (MSE) Loss. This will be used with the solver to
    // train the model.
    let mse_layer_cfg = LayerConfig::new("mse", LayerType::MeanSquaredError);
    regressor_cfg.add_layer(mse_layer_cfg);

    // Setup an Optimiser
    let solver_cfg = SolverConfig {
        minibatch_size: batch_size,
        base_lr: learning_rate,
        momentum,
        network: LayerConfig::new("network", net_cfg),
        objective: LayerConfig::new("regressor", regressor_cfg),
        ..SolverConfig::default()
    };

    Solver::from_config(backend.clone(), backend, &solver_cfg)
}

// Currently unused. It was supposed to be used for random tests with inlined
// verification or cross tests (Native <-> Cuda), but they aren't implemented
// yet.
pub fn uniformly_random_tensor<T, F>(_backend: &Backend<F>, dims: &[usize], low: T, high: T) -> SharedTensor<T>
where
    T: Copy + PartialEq + PartialOrd + rand::distributions::uniform::SampleUniform,
    F: IFramework,
    Backend<F>: IBackend,
{
    let dist = rand::distributions::Uniform::<T>::new_inclusive(low, high);
    let mut rng = rand::thread_rng();

    let mut xs = SharedTensor::new(&dims);
    {
        let native = get_native_backend();
        let native_dev = native.device();
        {
            let mem = xs.write_only(native_dev).unwrap();
            let mem_slice = mem.as_mut_slice::<T>();

            for x in mem_slice {
                *x = dist.sample(&mut rng);
            }
        }
        // not functional since, PartialEq has yet to be implemented for Device
        // but tbh this is test only so screw the extra dangling ununsed memory alloc
        //       let other_dev = backend.device();
        //       if other_dev != native_dev {
        //           xs.read(other_dev).unwrap();
        //           xs.drop_device(native_dev).unwrap();
        //       }
    }
    xs
}

pub fn write_to_tensor<T, F>(_backend: &Backend<F>, xs: &mut SharedTensor<T>, data: &[f64])
where
    T: ::std::marker::Copy + NumCast,
    F: IFramework,
    Backend<F>: IBackend,
{
    assert_eq!(xs.desc().size(), data.len());
    let native = get_native_backend();
    let native_dev = native.device();
    {
        let mem = xs.write_only(native_dev).unwrap();
        let mem_buffer = mem.as_mut_slice::<T>();
        for (i, x) in data.iter().enumerate() {
            mem_buffer[i] = cast::<_, T>(*x).unwrap();
        }
    }
    // not functional since, PartialEq has yet to be implemented for Device
    // but tbh this is test only so screw the extra dangling ununsed memory alloc
    //       let other_dev = backend.device();
    //       if other_dev != native_dev {
    //           xs.read(other_dev).unwrap();
    //           xs.drop_device(native_dev).unwrap();
    //       }
}

pub fn filled_tensor<T, F>(backend: &Backend<F>, dims: &[usize], data: &[f64]) -> SharedTensor<T>
where
    T: ::std::marker::Copy + NumCast,
    F: IFramework,
    Backend<F>: IBackend,
{
    let mut x = SharedTensor::new(&dims);
    write_to_tensor(backend, &mut x, data);
    x
}

fn main() {
    let _ = env_logger::builder()
        .default_format()
        .filter_level(log::LevelFilter::Trace)
        .try_init();

    eprintln!("XXXXXXXXXXXXXX");
    let cfg = sillynet();

    let backend = Rc::new(native_backend());
    let mut solver = add_solver::<Native>(backend.clone(), cfg, 1, 1., 1.);

    let mut input_lock = Arc::new(RwLock::new(uniformly_random_tensor(
        &backend,
        &[1, 2, 3, 2],
        0.0f32,
        256.0f32,
    )));
    let mut label_lock = Arc::new(RwLock::new(filled_tensor(&backend, &[1, 1], &[1.])));

    log::info!("Start training 1 trivial minibatch...");
    solver.train_minibatch(input_lock, label_lock);

    log::info!("Training complete;");
    let mut buf = Vec::<u8>::new();

    // save
    log::info!("Saving..");
    solver.mut_network().save(&mut buf).unwrap();
    log::info!("Saved;");

    log::info!("Loading..");
    // load the same
    let reincarnation = Layer::<Backend<Native>>::load(backend, &mut buf.as_slice()).unwrap();
    log::info!("Loaded;");
    log::info!("Cmp..");
    assert_eq!(solver.mut_network(), &reincarnation);
}

// fn foo() {
//     let p = "./foo.serialized.capnp";
//     {
//         let mut f = File::options().truncate(true).create(true).write(true).open(p).unwrap();
//         // let mut builder = juice::juice_capnp::sequential_config::Builder;
//         let mut builder = capnp::message::TypedBuilder::<juice::juice_capnp::sequential_config::Owned>::new_default();
//         let facade = &mut builder.get_root().unwrap();
//         cfg.write_capnp(facade);

//         capnp::serialize::write_message(&mut f, builder.borrow_inner()).unwrap();
//     }
//     let reincarnation = {
//         let f = File::options().read(true).open(p).unwrap();

//         let reader = BufReader::new(f);
//         let reader = capnp::serialize::try_read_message(
//             reader,
//             capnp::message::ReaderOptions {
//                 traversal_limit_in_words: None,
//                 nesting_limit: 100,
//             }).unwrap().unwrap();
//         <SequentialConfig as CapnpRead>::read_capnp(reader.get_root().unwrap())
//     };

//     assert_eq!(dbg!(cfg), dbg!(reincarnation));

// }
