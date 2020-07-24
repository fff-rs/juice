use csv::Reader;
use rand::prelude::*;

use crate::data_processing::tokenisation::{tokenize_label_input, tokenize_input};
use std::collections::HashMap;
use crate::data_processing::random_csv_access::RandomFileAccess;

pub type NetworkInput = (Option<Vec<f32>>, Vec<Vec<f32>>);

pub struct DataBackedLoader {
    access: RandomFileAccess,
    data_source: String
}

impl DataBackedLoader {
    pub(crate) fn from_path(data_source: String) -> DataBackedLoader {
        DataBackedLoader {
            access: RandomFileAccess::new(data_source.clone()),
            data_source
        }
    }
}

impl DataBackedLoader {
    /// Split Data Iterators in Training & Test Sets
    ///
    /// Training & Test Indexes are generated from rand at `holdout_percentage` indicating the
    /// percentage of indexes to be placed into the test set.
    ///
    /// Data is taken from a `FastFileAccess` of an input CSV by initially mapping the newline
    /// offsets in the file. This allows an initial O(N) read of the CSV, but an O(1) read of a
    /// single line, which is more efficient for random read, and doesn't require holding the entire
    /// file in memory at any one time.
    pub(crate) fn train_test_split (
        self,
        holdout_percentage: f32,
        input_columns: usize,
        label_columns: Option<usize>,
        batch_shape: usize)
        -> RawDataLoader {
        let data_backing : RandomFileAccess = RandomFileAccess::new(self.data_source.clone());
        let training_input_backend : RandomFileAccess = data_backing.clone();
        let training_label_backend : RandomFileAccess = data_backing.clone();
        let test_input_backend : RandomFileAccess = data_backing.clone();
        let test_label_backend : RandomFileAccess = data_backing.clone();
        let (train_index, test_index) = generate_holdout_indices(data_backing.len(), holdout_percentage);
        let training_label_stream = train_index.clone()
            .into_iter()
            .map( move |index| split_line_to_columns(training_label_backend.clone(), index, label_columns.unwrap_or(0), input_columns));

        let training_input_stream = train_index
            .into_iter()
            .map( move |index| split_line_to_columns(training_input_backend.clone(), index, label_columns.unwrap_or(0), input_columns));

        let test_label_stream = test_index.clone()
            .into_iter()
            .map( move |index| split_line_to_columns(test_label_backend.clone(), index, label_columns.unwrap_or(0), input_columns));

        let test_input_stream = test_index
            .into_iter()
            .map( move |index| split_line_to_columns(test_input_backend.clone(), index, label_columns.unwrap_or(0), input_columns));

        RawDataLoader {
            input_shape: vec![batch_shape, input_columns],
            output_shape: vec![batch_shape, label_columns.unwrap_or(0)],
            training_input_stream: Box::new(training_input_stream),
            training_label_stream: Box::new(training_label_stream),
            test_input_stream: Box::new(test_input_stream),
            test_label_stream: Box::new(test_label_stream)
        }
    }
}

pub struct RawDataLoader {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    training_input_stream: Box<dyn Iterator<Item=Vec<Vec<u8>>>>,
    training_label_stream: Box<dyn Iterator<Item=Vec<Vec<u8>>>>,
    test_input_stream: Box<dyn Iterator<Item=Vec<Vec<u8>>>>,
    test_label_stream: Box<dyn Iterator<Item=Vec<Vec<u8>>>>,
}

impl RawDataLoader {
    /// Create Tokens from Input
    ///
    /// Currently only supports an input shape of (batchSize, 1) and an output shape of (batchSize, 1).
    /// The stream input should contain up to two columns - an input and an output, in that order.
    pub(crate) fn tokenize(self,
                           label_tokenizer: HashMap<String, usize>,
                           phrase_length: usize) -> TokenLoader {
        assert_eq!(self.input_shape[1], 1);
        assert_eq!(self.output_shape[1], 1);
        let training_input_stream = tokenize_input(map_to_string(Box::new(self.training_input_stream), 0), phrase_length);
        let training_label_stream = tokenize_label_input(label_tokenizer.clone(), map_to_string(Box::new(self.training_label_stream), 1));
        let test_input_stream = tokenize_input(map_to_string(Box::new(self.test_input_stream), 0), phrase_length);
        let test_label_stream = tokenize_label_input(label_tokenizer, map_to_string(Box::new(self.test_label_stream), 1));
        TokenLoader {
            input_shape: self.input_shape,
            output_shape: self.output_shape,
            training_input_stream,
            training_label_stream,
            test_input_stream,
            test_label_stream
        }
    }
}

pub struct TokenLoader {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    training_input_stream: Box<dyn Iterator<Item=Vec<f32>>>,
    training_label_stream: Box<dyn Iterator<Item=f32>>,
    test_input_stream: Box<dyn Iterator<Item=Vec<f32>>>,
    test_label_stream: Box<dyn Iterator<Item=f32>>,
}

impl TokenLoader {
    pub(crate) fn train_test(self) -> (BatchedData, BatchedData) {
        (
            BatchedData {
                input_shape: self.input_shape.clone(),
                output_shape: self.output_shape.clone(),
                input_stream: self.training_input_stream,
                label_stream: self.training_label_stream
            },
            BatchedData {
                input_shape: self.input_shape,
                output_shape: self.output_shape,
                input_stream: self.test_input_stream,
                label_stream: self.test_label_stream
            }
        )
    }
}

pub struct BatchedData {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    input_stream: Box<dyn Iterator<Item=Vec<f32>>>,
    label_stream: Box<dyn Iterator<Item=f32>>,
}

impl BatchedData {
    pub fn len(&self) -> usize {
        self.input_shape[0]
    }

    pub fn empty(&self) -> bool {
        self.len() == 0
    }
}

impl Iterator for BatchedData {
    type Item = (Vec<Vec<f32>>, Option<Vec<f32>>);

    fn next(&mut self) -> Option<Self::Item> {
        let batch_size = self.input_shape[0];
        let mut input_buffer: Vec<Vec<f32>> = Vec::with_capacity(batch_size);
        let mut label_buffer: Vec<f32> = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            match (*self.input_stream).next() {
                Some(elem) => input_buffer.push(elem),
                None => break
            }
            match (*self.label_stream).next() {
                Some(elem) => label_buffer.push(elem),
                None => break
            }
        }
        Some((input_buffer, Some(label_buffer)))
    }
}

fn map_to_string(input: Box<dyn Iterator<Item=Vec<Vec<u8>>>>, column_index: usize) -> Box<dyn Iterator<Item=String>> {
    Box::new(input.map(move|item| item[column_index].iter().map(|bytes| *bytes as char).collect::<String>()))
}

fn split_line_to_columns(data_backing: RandomFileAccess, line_index: usize, label_columns: usize, input_columns: usize) -> Vec<Vec<u8>> {
    let backing : Vec<Vec<u8>> = data_backing
        .read_from_line(line_index)
        .split(|a| a == &44u8)
        .map(|x| x.to_vec())
        .collect();
    assert_eq!(backing.len(), label_columns + input_columns);
    backing
}

fn generate_holdout_indices(data_length: usize, holdout_split: f32) -> (Vec<usize>, Vec<usize>) {
    let mut train_index: Vec<usize> = Vec::with_capacity(data_length);
    let mut test_index: Vec<usize> = Vec::with_capacity(data_length);
    let mut rng = rand::thread_rng();
    for (row_index, in_train) in (0..data_length)
        .map(|_| rng.gen_range::<f32, f32, f32>(0.0, 1.0) > holdout_split)
        .enumerate()
    {
        if in_train {
            train_index.push(row_index)
        } else {
            test_index.push(row_index)
        }
    }
    return(train_index, test_index)
}