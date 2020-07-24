mod random_csv_access;
mod data_loader;
pub(crate) mod tokenisation;

use data_loader::{DataBackedLoader, TokenLoader};
use tokenisation::label_tokens;
use std::fs::File;
use csv::Reader;

pub(crate) fn stack_overflow_data(path: &str, shuffle: bool) -> (Vec<String>, Vec<String>) {
    let rdr = Reader::from_reader(File::open(path).unwrap());
    let input_data = rdr.into_deserialize().map(move |row| match row {
        Ok(value) => {
            let row_vec: Box<Vec<String>> = Box::new(value);
            let post = row_vec[0].clone();
            let tags = row_vec[1].clone();
            (post, tags)
        }
        _ => {
            println!("no value");
            panic!();
        }
    }).collect::<Vec<(String, String)>>();
    let label = input_data.clone().into_iter().map(|(_post, label)| label).collect();
    let post = input_data.into_iter().map(|(_label, post)| post).collect();
    return (label, post)
}


pub fn network_dataloader(holdout_percentage: f32, batch_size: usize, phrase_length: usize) -> TokenLoader {
    let path = "./data/stackdata.csv".to_string();
    let (labels, text_input) = stack_overflow_data(&path, true);
    let label_tokenizer = label_tokens(labels.clone());
    DataBackedLoader::from_path(path)
        .train_test_split(holdout_percentage, 1, Some(1), batch_size)
        .tokenize(label_tokenizer, phrase_length)
}