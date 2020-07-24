use coaster::frameworks::cuda::get_cuda_backend;
use coaster::prelude::*;
use coaster_nn::{DirectionMode, RnnInputMode, RnnNetworkMode};
use juice::layer::*;
use juice::layers::*;
use juice::solver::*;
use juice::util::*;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

use crate::model::params::PHRASE_LENGTH;

pub(crate) fn train_one_batch(
    network: &mut Solver<Backend<Cuda>, Backend<Cuda>>,
    batch_size: usize,
    inputs: Vec<Vec<f32>>,
    labels: Vec<f32>,
    evaluator: &mut ConfusionMatrix
) {
    let input = SharedTensor::<f32>::new(&[batch_size, PHRASE_LENGTH]);
    let input_lock = Arc::new(RwLock::new(input));

    let label = SharedTensor::<f32>::new(&[batch_size, 1]);
    let label_lock = Arc::new(RwLock::new(label));

    {
        let mut input_tensor = input_lock.write().unwrap();
        let mut label_tensor = label_lock.write().unwrap();
        write_batch(&mut input_tensor, &inputs.clone().into_iter().flatten().collect::<Vec<f32>>());
        write_batch(&mut label_tensor, &labels);
    }

    // Train the network
    println!("Training Minibatch");
    let inferred_out = network.train_minibatch(input_lock.clone(), label_lock.clone());
    println!("Completed Training Minibatch");
    let mut inferred = inferred_out.write().unwrap();
    let predictions = evaluator.get_predictions(&mut inferred);
    evaluator.add_samples(&predictions, &labels.clone().into_iter().map(|x| x as usize).collect::<Vec<usize>>());
    println!(
        "Last sample: {} | Accuracy {}",
        evaluator.samples().iter().last().unwrap(),
        evaluator.accuracy()
    );
}
