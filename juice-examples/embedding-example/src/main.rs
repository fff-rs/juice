extern crate env_logger;

mod data_processing;
mod model;

use clap::Clap;
use std::rc::Rc;
use std::fs::File;
use std::sync::{Arc, RwLock};

use coaster::frameworks::cuda::get_cuda_backend;
use coaster::prelude::*;
use coaster_nn::{DirectionMode, RnnInputMode, RnnNetworkMode};
use juice::layer::*;
use juice::layers::*;
use juice::solver::*;
use juice::util::*;

use data_processing::network_dataloader;
use model::network::create_network;
use model::train::train_one_batch;
use model::test::evaluate_batch;
use model::solver::add_solver;

use model::params::*;

/// This doc string acts as a help message when the user runs '--help'
/// as do all doc strings on fields
#[derive(Clap)]
#[clap(version = "1.0", author = "Lissa H <lissahyacinth@gmail.com>")]
struct Opts {
    /// Sets a custom config file. Could have been an Option<T> with no default too
    #[clap(short, long, default_value = "default.conf")]
    config: String,
    /// A level of verbosity, and can be used multiple times
    #[clap(short, long, parse(from_occurrences))]
    verbose: i32,
}

fn train_network(batch_size: Option<usize>, learning_rate: Option<f32>,
                 momentum: Option<f32>, epochs: Option<usize>,  holdout_percentage: Option<f32>, network_file: Option<File>) {
    // Initialise a CUDA Backend, and the CUDNN and CUBLAS libraries.
    let backend = Rc::new(get_cuda_backend());

    let batch_size = batch_size.unwrap_or(10);
    let learning_rate = learning_rate.unwrap_or(0.1f32);
    let momentum = momentum.unwrap_or(0.00f32);
    let epochs = epochs.unwrap_or(3usize);
    let holdout_percentage = holdout_percentage.unwrap_or(0.8);

    // Initialise a Sequential Layer
    let net_cfg = create_network(batch_size, PHRASE_LENGTH, OUTPUT_SIZE);
    let mut solver = add_solver(net_cfg, backend, batch_size, learning_rate, momentum);

    // Define Evaluation Method - Using Mean Squared Error
    let mut training_evaluator =
        ::juice::solver::ConfusionMatrix::new(OUTPUT_SIZE);
    // Indicate how many samples to average loss over
    training_evaluator.set_capacity(Some(2000));

    let mut test_evaluator =
        ::juice::solver::ConfusionMatrix::new(OUTPUT_SIZE);
    // Indicate how many samples to average loss over
    test_evaluator.set_capacity(Some(2000));

    for epoch in 0..epochs {
        println!("Epoch {}", epoch);
        let mut data_loader = network_dataloader(
            holdout_percentage,
            batch_size,
            PHRASE_LENGTH);
        let (mut training_loader, mut test_loader) = data_loader.train_test();
        while let Some((inputs, Some(labels))) = training_loader.next() {
            train_one_batch(&mut solver, batch_size, inputs, labels, &mut training_evaluator);
        }
        //evaluate_batch(&mut solver,  batch_size, test_loader);
    }

    // Write the network to a file
    // FIXME: Resolve File Issues
    /*if let Some(f) = network_file {
        solver.mut_network().save(f).unwrap();
    }*/
}

fn main() {
    env_logger::init();
    train_network(Some(125), Some(0.01), None, Some(2), Some(0.8), None);
}
