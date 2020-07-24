//! Utility layer to modify an input to an embedding[1]
//!
//! Word or Token Input Indices are translated to weight vectors, which are then trained against
//! the output of the network.
//!
//! i.e.
//! Input - ['the', 'cat', 'sat', 'on', 'the', 'mat']
//! would be translated into the indices
//! Indices - [0, 1, 2, 3, 0, 4]
//! Each index is used to lookup the relevant weight embedding, i.e.
//! Word Embeddings with Vocab 5, and Dimension 3;
//! [(0.2, 0.3, 0.1), (0.1, 0.3, 0.2), (0.4, 0.1, 0.2), (0.5, 0.1, 0.3), (0.2, 0.3, 0.1), (0.2, 0.2, 0.2)]
//! The weight for 'the' appears twice, but all others are initialised randomly. This is because the
//! word embedding layer is only retrieving the relevant weight for that text, and not generating
//! new potential weights for each time the word occurs.
//!
//! While this example discusses words, embeddings may also be generated from subwords using tokenizers
//! like SentenceWords, which breaks words into likely subwords for the relevant language, i.e.
//! the subwords of gymnasium are likely to be gym, nas, and ium - gym being an independent word, nas
//! showing up in other words (nasty, nascent, etc), and ium being a common suffix. The Embedding is
//! then not counting weights for the words, but the subwords.
//! Some subword generators will also include if the word is an infix - occurs within a word - a
//! prefix, suffix, or entire word, by including the spacing that commonly surrounds it. 'nas' - our
//! infix in Gymnasium, is more likely to occur as '_nas', indicating it begins at the start of a word.
//! [1]:https://en.wikipedia.org/wiki/Word_embedding

use crate::capnp_util::*;
use crate::co::{IBackend, SharedTensor};
use crate::coblas::transpose::Transpose;
use crate::juice_capnp::embedding_config as capnp_config;
use crate::layer::*;
use crate::util::{LayerOps, native_backend};
use crate::util::{native_scalar, ArcLock};
use crate::weight::FillerType;

// TODO: Remove when Embedding Passes
use crate::util::write_to_memory;

#[derive(Debug)]
/// Embedding Layer
pub struct Embedding {
    /// Dimension size for embedding vectors
    embedding_dimension: usize,
    /// Number of embedding tokens included
    vocab_size: usize,
    /// Number of tokens per instance/phrase
    phrase_length: usize,

    one: SharedTensor<f32>,
    zero: SharedTensor<f32>,
}

fn one_hot_vector(inp: Vec<f32>, vec_size: usize) -> Vec<f32> {
    let mut ohe_vec = Vec::with_capacity(vec_size * inp.len());

    for inp_elem in inp {
        let mut full_vec = vec![0_f32; vec_size];
        full_vec[inp_elem as usize] = inp_elem;
        ohe_vec.append(&mut full_vec);
    }
    ohe_vec
}


impl Embedding {
    /// Create a Embedding layer from a EmbeddingConfig.
    pub fn from_config(config: &EmbeddingConfig) -> Embedding {
        let one = native_scalar(1f32);
        let zero = native_scalar(0f32);

        Embedding {
            embedding_dimension: config.embedding_dimension,
            vocab_size: config.vocab_size,
            phrase_length: config.phrase_length,

            one,
            zero,
        }
    }
}

impl<B: conn::Gather<f32> + LayerOps<f32> + IBackend> ILayer<B> for Embedding {
    fn auto_weight_blobs(&self) -> bool {
        true
    }

    fn reshape(
        &mut self,
        backend: ::std::rc::Rc<B>,
        input_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
        input_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
        weights_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
        weights_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
        output_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
        output_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
    ) {
        let input = input_data[0].read().unwrap();
        let input_shape = input.desc();

        // Input should be [Batch, Embedding]
        assert_eq!(input_shape.len(), 2);
        let batch_size = input_shape[0];
        let output_shape = (batch_size, self.embedding_dimension, self.phrase_length);
        let weight_shape = (self.embedding_dimension, self.vocab_size);

        output_data[0].write().unwrap().resize(&output_shape).unwrap();
        output_gradient[0].write().unwrap().resize(&output_shape).unwrap();

        if let Some(weight) = weights_data.get(0) {
            weight.write().unwrap().resize(&weight_shape).unwrap();
            let filler = FillerType::Glorot {
                input_size: self.phrase_length,
                output_size: batch_size * self.phrase_length * self.embedding_dimension,
            };
            filler.fill(&mut weight.write().unwrap());
        }

        if let Some(weight) = weights_data.get(1) {
            weight.write().unwrap().resize(&weight_shape).unwrap();
            let filler = FillerType::Constant {
                value: 0.0,
            };
            filler.fill(&mut weight.write().unwrap());
        }

        if let Some(weight) = weights_gradient.get(0) {
            weight.write().unwrap().resize(&weight_shape).unwrap();
        }
        if let Some(weight) = weights_gradient.get(1) {
            weight.write().unwrap().resize(&weight_shape).unwrap();
            let filler = FillerType::Constant {
                value: 0.0,
            };
            filler.fill(&mut weight.write().unwrap());
        }
    }
}

impl<B: conn::Gather<f32> + IBackend> ComputeOutput<f32, B> for Embedding {
    fn compute_output(
        &self,
        backend: &B,
        weights: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        output_data: &mut [&mut SharedTensor<f32>],
    ) {
        let src = input_data[0];
        let src_desc = src.desc();
        let batch_size = src_desc[0];

        let weight = weights[0];
        let dest = &mut output_data[0];

        backend
            .gather(
                src,
                weight,
                *dest,
                self.embedding_dimension,
                self.phrase_length,
                self.vocab_size,
                batch_size,
            )
            .unwrap();

        let native = native_backend();
        let src_data : Vec<f32> = (dest
            .read(native.device())
            .unwrap()
            .as_slice::<f32>()).to_vec();
    }
}

impl<B: IBackend + LayerOps<f32>> ComputeInputGradient<f32, B> for Embedding {
    fn compute_input_gradient(
        &self,
        backend: &B,
        weights_data: &[&SharedTensor<f32>],
        output_data: &[&SharedTensor<f32>],
        output_gradients: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        input_gradients: &mut [&mut SharedTensor<f32>],
    ) {
        let native = native_backend();
        //let desc = input_gradients[0].desc().iter().fold(1, |acc, x| acc * x);
        //write_to_memory(&mut input_gradients[0].write_only(native.device()).unwrap(), &vec![0.0f32; desc]);
    }
}

impl<B: IBackend + conn::BatchedStridedSum<f32> + LayerOps<f32>> ComputeParametersGradient<f32, B> for Embedding {
    fn compute_parameters_gradient(
        &self,
        backend: &B,
        output_data: &[&SharedTensor<f32>],
        output_gradients: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        parameters_gradients: &mut [&mut SharedTensor<f32>],
    ) {
        debug!("Calculating Input Gradient for Embedding");

        let src = input_data[0];
        let src_desc = src.desc();
        let batch_size = src_desc[0];
        let dest = &mut parameters_gradients[0];
        //dbg!(dest.desc());
        let native = native_backend();

        let src_data : Vec<f32> = (src
            .read(native.device())
            .unwrap()
            .as_slice::<f32>()).to_vec();

        let ohe_src_data = one_hot_vector(src_data, self.vocab_size);
        // Batch Size; Phrase Length; Vocab Size
        let mut ohe_input = SharedTensor::new(&[batch_size, src_desc[1], self.vocab_size]);
        write_to_memory(&mut ohe_input.write_only(native.device()).unwrap(), &ohe_src_data);
        // TODO: Implement a CPU version of OHE from R in Rust, and test if this is
        // sufficient to make the network learn.

        // FIXME: The current output was assumed to be 3x2, but it's actually the weights, VocabxEmbed. or 3x10k.
        // Assuming that'll break this, somehow. :) It does! GEMM Batched No Longer works, likely due to the
        // changes made earlier. lda 10, ldb 10k, ldc 3. Error is with LDC, it should be 10k not 3.
        //
        let mut batch_stride_out = SharedTensor::new(&[batch_size, self.embedding_dimension, self.vocab_size]);

        let out_grad : Vec<f32> = (output_gradients[0]
            .read(native.device())
            .unwrap()
            .as_slice::<f32>()).to_vec();
        //dbg!(out_grad);
        //dbg!(ohe_src_data.clone());

        backend
            .gemm_batched(
                &self.one,
                Transpose::NoTrans,
                output_gradients[0],
                Transpose::Trans,
                &ohe_input,
                &self.zero,
                &mut batch_stride_out,
                batch_size
            )
            .unwrap();

        let batch_stride_out_data : Vec<f32> = (batch_stride_out
            .read(native.device())
            .unwrap()
            .as_slice::<f32>()).to_vec();

        //dbg!(batch_stride_out_data.clone());

        backend.batched_strided_sum(
            &batch_stride_out,
            parameters_gradients[0],
            batch_size,
            3,
            10000
        ).unwrap();

        /*dbg!(parameters_gradients[0]
            .read(native.device())
            .unwrap()
            .as_slice::<f32>());*/
    }
}

#[derive(Debug, Clone)]
/// Specifies configuration parameters for a Embedding Layer.
pub struct EmbeddingConfig {
    pub embedding_dimension: usize,
    pub vocab_size: usize,
    pub phrase_length: usize,
}

impl<'a> CapnpWrite<'a> for EmbeddingConfig {
    type Builder = capnp_config::Builder<'a>;

    /// Write the EmbeddingConfig into a capnp message.
    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder
            .reborrow()
            .set_embedding_dimension(self.embedding_dimension as u64);
        builder.reborrow().set_vocab_size(self.vocab_size as u64);
        builder.reborrow().set_phrase_length(self.phrase_length as u64);
    }
}

impl<'a> CapnpRead<'a> for EmbeddingConfig {
    type Reader = capnp_config::Reader<'a>;

    fn read_capnp(reader: Self::Reader) -> Self {
        let embedding_dimension = reader.get_embedding_dimension() as usize;
        let vocab_size = reader.get_vocab_size() as usize;
        let phrase_length = reader.get_phrase_length() as usize;
        EmbeddingConfig {
            embedding_dimension,
            vocab_size,
            phrase_length,
        }
    }
}

impl Into<LayerType> for EmbeddingConfig {
    fn into(self) -> LayerType {
        LayerType::Embedding(self)
    }
}
