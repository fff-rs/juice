use std::collections::HashMap;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::tokenizer::{Tokenizer, Trainer};

pub(crate) fn generate_encoding(vocab_size: usize) {
    let trainer: Box<dyn Trainer> = Box::new(
        BpeTrainerBuilder::new()
            .show_progress(false)
            .vocab_size(vocab_size)
            .min_frequency(0)
            .build(),
    );

    let mut tokenizer = Tokenizer::new(Box::new(BPE::default()));
    tokenizer.with_normalizer(Box::new(Sequence::new(vec![
        Box::new(Strip::new(true, true)),
        Box::new(NFC),
    ])));
    tokenizer.with_pre_tokenizer(Box::new(ByteLevel::default()));

    // This is using the general form of train, which replaces the self.model
    // with a custom model.
    tokenizer
        .train(&trainer, vec!["./data/text_input.txt".to_string()])
        .unwrap();
    tokenizer
        .save("./tokenizers/bpe.model".as_ref(), true)
        .unwrap();
}

fn load_bpe_model() -> Tokenizer {
    Tokenizer::from_file("./tokenizers/bpe.model").unwrap()
}

pub(crate) fn label_tokens(labels: Vec<String>) -> HashMap<String, usize> {
    let mut unique_labels = labels.clone();
    unique_labels.sort();
    unique_labels.dedup();
    unique_labels
        .into_iter()
        .enumerate()
        .map(|(x, y)| (y, x))
        .collect()
}

pub(crate) fn tokenize_input(
    input: Box<dyn Iterator<Item = String>>,
    input_length: usize,
) -> Box<dyn Iterator<Item = Vec<f32>>> {
    let bpe_model = load_bpe_model();
    Box::new(input.map(move |batched_input| {
        let encoding = bpe_model.encode(batched_input, false).unwrap();
        let mut tokenised_input = encoding
            .get_ids()
            .iter()
            .map(|id| *id as f32)
            .collect::<Vec<f32>>();
        match tokenised_input.len().cmp(&input_length) {
            std::cmp::Ordering::Less => {
                let diff = input_length - tokenised_input.len();
                tokenised_input.append(&mut vec![0.0; diff]);
                tokenised_input
            }
            std::cmp::Ordering::Greater => {
                tokenised_input[0..input_length].to_vec()
            }
            std::cmp::Ordering::Equal => tokenised_input,
        }
    }))
}

pub(crate) fn tokenize_batched_input(
    input: Box<dyn Iterator<Item = Vec<String>>>,
    input_length: usize,
) -> Box<dyn Iterator<Item = Vec<Vec<f32>>>> {
    let bpe_model = load_bpe_model();
    Box::new(input.map(move |batched_input| {
        bpe_model
            .encode_batch(batched_input, false)
            .unwrap()
            .iter()
            .map(|encoding| {
                let mut tokenised_input = encoding
                    .get_ids()
                    .iter()
                    .map(|id| *id as f32)
                    .collect::<Vec<f32>>();
                match tokenised_input.len().cmp(&input_length) {
                    std::cmp::Ordering::Greater => {
                        let diff = tokenised_input.len() - input_length;
                        tokenised_input.append(&mut vec![0.0; diff]);
                        tokenised_input
                    }
                    std::cmp::Ordering::Less => {
                        let diff = input_length - tokenised_input.len();
                        tokenised_input[0..diff].to_vec()
                    }
                    std::cmp::Ordering::Equal => tokenised_input,
                }
            })
            .collect::<Vec<Vec<f32>>>()
    }))
}

pub(crate) fn tokenize_label_input(
    label_tokenizer: HashMap<String, usize>,
    labels: Box<dyn Iterator<Item = String>>,
) -> Box<dyn Iterator<Item = f32>> {
    Box::new(labels.map(move |label| label_tokenizer[&label] as f32))
}