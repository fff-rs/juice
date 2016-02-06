//! Provides the generics and interfaces for the specific [Layers][layers].
//! [layers]: ../layers/index.html
use co::IBackend;
use co::SharedTensor;
use co::plugin::numeric_helpers::Float;
use coblas::plugin::Dot;
use shared_memory::ArcLock;
use layers::*;
use weight::WeightConfig;
use util::{native_backend, LayerOps};
use std::fmt;
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::{Arc, RwLock};

#[derive(Debug)]
/// The generic Layer
pub struct Layer<B: IBackend + LayerOps<f32>> {
    /// Identifies the Network
    ///
    /// The name is mainly used for logging purposes.
    pub name: String,
    /// The configuration of the Layer
    pub config: Box<LayerConfig>,
    /// The [implementation][1] of the Layer.
    /// [1]: ../layers/index.html
    ///
    /// This is the part that does most of the work ([forward][2]/[backward][3]).
    /// [2]: ./trait.ILayer.html#method.forward
    /// [3]: ./trait.ILayer.html#method.backward
    pub worker: Box<ILayer<B>>,

    backend: Rc<B>,

    /// Determines if layer will skip comutations for [backward][1] step.
    /// [1]: ./trait.ILayer.html#method.backward
    needs_backward: bool,

    /// The vector that stores shared references to the weights in the form of blobs.
    pub weights_data: Vec<ArcLock<SharedTensor<f32>>>,
    /// The vector that stores shared references to the weights in the form of blobs.
    pub weights_gradient: Vec<ArcLock<SharedTensor<f32>>>,
    // contains all the learnable weights (does not include bias(?) and shared weights)
    learnable_weights: Vec<ArcLock<SharedTensor<f32>>>,
    // learning rate for each weight
    weights_lr: Vec<Option<f32>>,
    // weight decay for each weight
    weights_weight_decay: Vec<Option<f32>>,
    // display name for each weight
    weights_display_names: Vec<String>,

    /// Vector indicating whether to compute the diff of each weight blob.
    ///
    /// You can safely ignore false values and always compute gradients
    /// for all weights, but possibly with wasteful computation.
    ///
    /// Can be used by some [Layer implementations][1] to optimize performance.
    /// [1]: ../layers/index.html
    weight_propagate_down: Vec<bool>,

    /// References to all the bottom blobs of the layer.
    pub bottom_blobs_data: Vec<ArcLock<SharedTensor<f32>>>,
    /// References to all the bottom blobs of the layer.
    pub bottom_blobs_gradient: Vec<ArcLock<SharedTensor<f32>>>,
    bottom_blob_names: Vec<String>,
    bottom_need_backwards: Vec<bool>,

    /// References to all the top blobs of the layer.
    pub top_blobs_data: Vec<ArcLock<SharedTensor<f32>>>,
    /// References to all the top blobs of the layer.
    pub top_blobs_gradient: Vec<ArcLock<SharedTensor<f32>>>,
    top_blob_names: Vec<String>,
    /// The vector that indicates whether each top blob contributes to
    /// the [loss][1] of the network and with which weight.
    /// [1]: http://caffe.berkeleyvision.org/tutorial/loss.html
    loss: Vec<f32>,

    /// All the blobs of the layer that can be addressed by name.
    ///
    /// Does not contain anonymous blobs.
    pub blob_names: HashMap<String, (ArcLock<SharedTensor<f32>>, ArcLock<SharedTensor<f32>>)>,
}

impl<B: IBackend + LayerOps<f32> + 'static> Layer<B> {
    /// Creates a new Layer from a [LayerConfig][1].
    /// [1]: ./struct.LayerConfig.html
    ///
    /// Used during [Network][2] initalization.
    ///
    /// [2]: ../network/struct.Network.html
    pub fn from_config(backend: Rc<B>, config: &LayerConfig) -> Layer<B> {
        let cl = config.clone();
        let cfg = Box::<LayerConfig>::new(cl);
        Layer {
            name: cfg.name.clone(),

            needs_backward: true,

            weights_data: Vec::new(),
            weights_gradient: Vec::new(),
            learnable_weights: Vec::new(),
            weight_propagate_down: Vec::new(),
            weights_lr: Vec::new(),
            weights_weight_decay: Vec::new(),
            weights_display_names: Vec::new(),

            bottom_blobs_data: Vec::new(),
            bottom_blobs_gradient: Vec::new(),
            bottom_blob_names: Vec::new(),
            bottom_need_backwards: Vec::new(),

            top_blobs_data: Vec::new(),
            top_blobs_gradient: Vec::new(),
            top_blob_names: Vec::new(),
            loss: vec![1f32, 1f32, 1f32],

            blob_names: HashMap::new(),

            backend: backend,

            worker: Layer::<B>::worker_from_config(&cfg),
            config: cfg,
        }
    }

    /// Helper for [from_config] to match a [LayerType][2] to its [implementation][3].
    /// [1]: #method.from_config
    /// [2]: ./enum.LayerType.html
    /// [3]: ../layers/index.html
    fn worker_from_config(config: &LayerConfig) -> Box<ILayer<B>> {
        match config.layer_type.clone() {
            LayerType::Convolution(layer_config) => Box::new(Convolution::from_config(&layer_config)),
            LayerType::FullyConnected(layer_config) => Box::new(FullyConnected::from_config(&layer_config)),
            LayerType::Pooling(layer_config) => Box::new(Pooling::from_config(&layer_config)),
            LayerType::ReLU => Box::new(ReLU),
            LayerType::Sigmoid => Box::new(Sigmoid),
            LayerType::SoftmaxLoss => Box::new(SoftmaxLoss::default()),
        }
    }

    /// Connect layer to the other layers in a [Network][1] and set up Blobs.
    /// [1]: ../network/struct.Network.html
    ///
    /// Connects to the bottoms provided by other layers via the `registry`.
    /// Adds top blobs to the layer and then adds them to the `registry`, so the next
    /// layers can connect them as their bottoms.
    /// In the end it intializes the underlying [layer implementation][2].
    ///
    /// [2]: ./trait.ILayer.html
    ///
    /// Called during [Network][1] initialization.
    pub fn connect(
        &mut self,
        registry: &mut HashMap<String, (ArcLock<SharedTensor<f32>>, ArcLock<SharedTensor<f32>>)>,
        weight_registry: &mut HashMap<String, (ArcLock<SharedTensor<f32>>, ArcLock<SharedTensor<f32>>, Option<f32>, Option<f32>)>) {
        // connect to all required bottoms
        for bottom_name in &self.config.bottoms.clone() {
            self.connect_bottom(bottom_name, registry)
        }
        // setup tops
        for (top_id, _) in self.config.tops.clone().iter().rev().enumerate() {
            self.append_top(top_id, registry);
        }
        let config = self.config.clone();
        for (top_id, _) in self.config.tops.clone().iter().rev().enumerate() {
            self.append_weight(&config, weight_registry, 0, top_id);
        }

        // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
        // specified fewer than the required number (as specified by
        // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
        let auto_top_blobs = self.worker.auto_top_blobs();
        debug!("Layer {} - auto_top_blobs: {}", &self.name, &auto_top_blobs);
        let min_top_blobs = self.worker.min_top_blobs();
        let exact_num_top_blobs = self.worker.exact_num_top_blobs();
        if auto_top_blobs {
            let needed_num_top = cmp::max(min_top_blobs, exact_num_top_blobs);
            for _ in 0..(needed_num_top - self.top_blobs_data.len()) {
                // Add "anonymous" top blobs -- do not add to registry
                // as we don't want these blobs to be usable as input
                // to other layers.
                info!("Adding anonymous top blob for layer {}", &self.name);
                self.create_anonymous_top();
            }
        }

        self.worker.init(self.backend.clone());
        self.worker.reshape(self.backend.clone(),
                            &self.bottom_blobs_data,
                            &mut self.weights_data,
                            &mut self.weights_gradient,
                            &mut self.top_blobs_data,
                            &mut self.top_blobs_gradient);
        for t in &self.top_blobs_data {
            println!("{} top shape: {:?}", self.name, t.read().unwrap().desc());
        }
    }

    /// Append blob as [bottom blob][1] to the Layer.
    /// [1]: ../layer/index.html
    ///
    /// During network initalization the blobs will be appended to the Layers as per their
    /// [LayerConfig][3]. It is also determined if a bottom blob skips backpropagation
    /// from [LayerConfig.propagate_down][3] (see also [init_backprop][5]).
    ///
    /// [3]: ../layer/struct.LayerConfig.html
    /// [5]: #method.init_backprop
    fn connect_bottom(&mut self, blob_name: &str, available_blobs: &mut HashMap<String, (ArcLock<SharedTensor<f32>>, ArcLock<SharedTensor<f32>>)>) {
        let bottom_id = self.config.bottoms.iter().position(|bottom_name| bottom_name == blob_name).unwrap();

        if !available_blobs.contains_key(&*blob_name) {
            error!("Unknown bottom blob {} (layer '{}', bottom_id: {})",
                   blob_name,
                   self.name,
                   bottom_id);
        }
        info!("{:<15} -> {:>15}", blob_name, self.name);

        self.bottom_blob_names.push(blob_name.to_owned());
        self.bottom_blobs_data.push(available_blobs[&*blob_name].0.clone());
        self.bottom_blobs_gradient.push(available_blobs[&*blob_name].1.clone());
        available_blobs.remove(&*blob_name);

        let mut propagate_down = true;
        // Check if the backpropagation on bottom_id should be skipped
        if !self.config.propagate_down.is_empty() {
            propagate_down = self.config.propagate_down[bottom_id];
        }
        let need_backward = propagate_down;
        self.bottom_need_backwards.push(need_backward);
    }

    /// Append blob as [top blob][1] to the Layer.
    /// [1]: ../layer/index.html
    ///
    /// During network initalization the blobs will be appended to the Layers as per their
    /// [LayerConfig][2]. It is also determined if computations can be done in-place, in which
    /// no additional Blob will be allocated.</br>
    /// Finally, the new blob will be added to the registry, so that the other layers can
    /// connect it as their bottom.
    /// [2]: ../layer/struct.LayerConfig.html
    fn append_top(&mut self,
                  top_id: usize,
                  registry: &mut HashMap<String, (ArcLock<SharedTensor<f32>>, ArcLock<SharedTensor<f32>>)>) {
        let layer_config = &self.config;

        let blob_name = layer_config.top(top_id).unwrap().clone();
        let blob_data: ArcLock<SharedTensor<f32>>;
        let blob_gradient: ArcLock<SharedTensor<f32>>;

        if layer_config.bottom(top_id).is_some() && *layer_config.bottom(top_id).unwrap() == blob_name {
            info!("{} -> {} (in-place)", layer_config.name, blob_name);
            blob_data = registry[&blob_name].0.clone();
            blob_gradient = registry[&blob_name].1.clone();
        } else if registry.contains_key(&blob_name) {
            // If we are not doing in-place computation but have duplicated blobs, raise an
            // error.
            error!("Top blob {} produced by multiple sources.", blob_name);
            return
        } else {
            {
                info!("{:<15} -> {:>15}", self.name, blob_name);
                info!("Input {} -> {}", top_id, blob_name);
            }

            let backend: Rc<IBackend<F=B::F>> = self.backend.clone();
            blob_data = Arc::new(RwLock::new(SharedTensor::new(backend.device(), &vec![1,1,1]).unwrap())); // [1,1,1] for CUDA
            blob_gradient = Arc::new(RwLock::new(SharedTensor::new(backend.device(), &vec![1,1,1]).unwrap())); // [1,1,1] for CUDA
        }
        self.top_blob_names.push(blob_name.clone());
        self.top_blobs_data.push(blob_data.clone());
        self.top_blobs_gradient.push(blob_gradient.clone());
        self.blob_names.insert(blob_name.clone(), (blob_data.clone(), blob_gradient.clone()));
        registry.insert(blob_name.clone(), (blob_data.clone(), blob_gradient.clone()));
    }

    /// Append anonymous blob as [top blob][1] to the Layer.
    /// [1]: ../layer/index.html
    ///
    /// [Layer implementations][2] may request creation of anonymous top blobs
    /// via [auto_top_blobs][3]. Since the blobs are not named, other layers can
    /// not use them as their bottom blobs.
    /// [2]: ./trait.ILayer.html
    /// [3]: ./trait.ILayer.html#method.auto_top_blobs
    fn create_anonymous_top(&mut self) {
        let blob_name = "(automatic)".to_owned();

        info!("{} -> {}", self.name, blob_name);

        let backend: Rc<IBackend<F=B::F>> = self.backend.clone();
        let top_data = Arc::new(RwLock::new(SharedTensor::new(backend.device(), &vec![1,1,1]).unwrap())); // [1,1,1] for CUDA
        let top_gradient = Arc::new(RwLock::new(SharedTensor::new(backend.device(), &vec![1,1,1]).unwrap())); // [1,1,1] for CUDA
        self.top_blobs_data.push(top_data);
        self.top_blobs_gradient.push(top_gradient);
    }

    fn append_weight(&mut self, layer_config: &LayerConfig, registry: &mut HashMap<String, (ArcLock<SharedTensor<f32>>, ArcLock<SharedTensor<f32>>, Option<f32>, Option<f32>)>, layer_id: usize, weight_id: usize) {
        info!("Appending weight to layer {}", &layer_config.name);
        let weights_len = self.weights_data.len();
        let weight_name = if weights_len > weight_id {
            layer_config.param(weight_id).unwrap().name.clone()
        } else {
            "".to_owned()
        };

        // use weight_name (or weight_id as a fallback) as display_name
        let display_name = if !weight_name.is_empty() {
            weight_name.clone()
        } else {
            format!("{}", weight_id)
        };
        self.weights_display_names.push(display_name.clone());
        // create name for registry
        let registry_name = format!("SHARED_WEIGHT_{}", display_name);

        // add to tracking vectors
        let net_weight_id = weights_len;
        let top_data = self.top_blobs_data[weight_id].read().unwrap();
        let weight_data = Arc::new(RwLock::new(SharedTensor::<f32>::new(top_data.latest_device(), top_data.desc()).unwrap()));
        let weight_gradient = Arc::new(RwLock::new(SharedTensor::<f32>::new(top_data.latest_device(), top_data.desc()).unwrap()));
        self.weights_data.push(weight_data.clone());
        self.weights_gradient.push(weight_gradient.clone());

        let mut weight_config = &WeightConfig::default();
        if layer_config.params_len() > weight_id {
            weight_config = layer_config.param(weight_id).unwrap();
        }
        // This layer "owns" this weight blob -- it is either anonymous
        // (i.e., not given a weight_name) or explicitly given a name that we
        // haven't already seen.
        if weight_name.is_empty() || !registry.contains_key(&registry_name) {
            // self.weight_owners.push(None);
            if !weight_name.is_empty() {
                registry.insert(weight_name.clone(),
                    (weight_data.clone(), weight_gradient.clone(), weight_config.lr_mult.clone(), weight_config.decay_mult.clone()));
            }
            let learnable_weight_id = self.learnable_weights.len();
            self.learnable_weights.push(weight_data.clone());
            // self.learnable_weight_ids.push(learnable_weight_id);
            self.weights_lr.push(weight_config.lr_mult.clone());
            self.weights_weight_decay.push(weight_config.decay_mult.clone());
        } else {
            // Named weight blob with name we've seen before: share weights

            let (shared_weight_data, shared_weight_gradient, shared_lr, shared_decay_mult) = registry.get(&registry_name).unwrap().clone();
            info!("Sharing weight blob '{}'", weight_name.clone());

            // TODO: move shape checking into reshape?
            // can only share weights if blobs match by shape or capacity
            // if weights_len > weight_id {
            //     if let Err(e) = layer_config.param(weight_id)
            //                                 .unwrap()
            //                                 .check_dimensions(&this_blob.read().unwrap(),
            //                                                   &owner_blob.read().unwrap(),
            //                                                   weight_name.clone(),
            //                                                   self.layers[owner_layer_id].name.clone(),
            //                                                   self.layers[layer_id].name.clone()) {
            //         error!("{}", e)
            //     }
            // }

            // can only share parameters if both have same lr_mult
            if let Some(lr_mult) = weight_config.lr_mult {
                if let Some(owner_lr_mult) = shared_lr {
                    if !lr_mult.eq(&owner_lr_mult) {
                        error!("Shared param '{}' has mismatched lr_mult.",
                               weight_name.clone());
                    }
                } else {
                    // this is the first shared instance that has a lr_mult value so we take that
                    registry.remove(&registry_name).unwrap();
                    registry.insert(registry_name.clone(), (shared_weight_data.clone(), shared_weight_gradient.clone(), weight_config.lr_mult, shared_decay_mult));
                }
            }
            // can only share weights if both have same decay_mult
            if let Some(decay_mult) = weight_config.decay_mult {
                if let Some(owner_decay_mult) = shared_decay_mult {
                    if !decay_mult.eq(&owner_decay_mult) {
                        error!("Shared param '{}' has mismatched decay_mult.",
                               weight_name.clone());
                    }
                } else {
                    // this is the first shared instance that has a decay_mult value so we take that
                    registry.remove(&registry_name).unwrap();
                    registry.insert(registry_name, (shared_weight_data.clone(), shared_weight_gradient.clone(), shared_lr, weight_config.decay_mult));
                }
            }
        }
    }

    /// Initializes layer for [backpropagation][1]
    /// [1]: https://en.wikipedia.org/wiki/Backpropagation
    ///
    /// Go through all the blobs of a layer to determine which blobs contribute to the
    /// loss of the next layer. We can skip backward computation for blobs that don't contribute
    /// to the loss.
    /// If all of the blobs skip backpropagation we set a flag to skip backpropagation
    /// of the whole layer.
    pub fn init_backprop(&mut self,
                     blobs_under_loss: &mut HashSet<String>,
                     blobs_skip_backp: &mut HashSet<String>) {
        let mut layer_contributes_loss = false;
        let mut layer_skip_propagate_down = true;
        for (top_id, top_blob) in self.top_blobs_data.iter().enumerate() {
            let blob_name = self.top_blob_names.get(top_id);

            // layer is a loss layer or under a loss layer
            if self.loss(top_id).is_some() || blob_name.is_some() && blobs_under_loss.contains(blob_name.unwrap()) {
                layer_contributes_loss = true;
            }
            // layer is not marked to skip backpropagation
            if blob_name.is_none() || blob_name.is_some() && !blobs_skip_backp.contains(blob_name.unwrap()) {
                layer_skip_propagate_down = false;
            }
            // layer contributes loss to some
            if layer_contributes_loss && !layer_skip_propagate_down {
                break;
            }
        }

        // If this layer can skip backward computation, also all his bottom blobs
        // don't need backpropagation
        if self.needs_backward && layer_skip_propagate_down {
            self.needs_backward = false;
            for (bottom_id, _) in self.bottom_blobs_data.iter().enumerate() {
                self.bottom_need_backwards[bottom_id] = false;
            }
        }
        // layer doesn't contribute loss so it does not need to be backpropagated
        if !layer_contributes_loss {
            self.needs_backward = false;
        }
        {
            info!("{} needs backward computation: {}",
                  self.name,
                  self.needs_backward);
        }

        for (bottom_id, bottom_name) in self.bottom_blob_names.iter().enumerate() {
            if layer_contributes_loss {
                blobs_under_loss.insert(bottom_name.clone());
            } else {
                self.bottom_need_backwards[bottom_id] = false;
            }
            if !self.bottom_need_backwards[bottom_id] {
                blobs_skip_backp.insert(bottom_name.clone());
            }
        }
    }

    /// Set [backpropagation][1] flags to force this layer to backpropagate.
    /// [1]: https://en.wikipedia.org/wiki/Backpropagation
    ///
    /// Is executed during Network initalization if [NetworkConfig][2].force_backward is true.
    /// Forcing backpropagation is useful for debugging.
    pub fn init_force_backward(&mut self) {
        self.needs_backward = true;
        for (bottom_id, _) in self.bottom_need_backwards.clone().iter().enumerate() {
            self.bottom_need_backwards[bottom_id] =
                *self.bottom_need_backwards
                     .get(bottom_id)
                     .unwrap_or(&self.worker.allow_force_backward(bottom_id));
        }
        for (weight_id, _) in self.weights_data.clone().iter().enumerate() {
            self.set_weight_propagate_down(weight_id, true);
        }
    }

    /// Uses the underlying layer implementation to compute a forward step.
    ///
    /// See [ILayer.forward](./trait.ILayer.html#method.forward)
    pub fn forward(&mut self) -> f32 {
        debug!("LAYER: {:?}", &self.name);
        self.worker.sync(&self.backend,
                         &mut self.bottom_blobs_data, &mut self.bottom_blobs_gradient,
                         &mut self.weights_data, &mut self.weights_gradient,
                         &mut self.top_blobs_data, &mut self.top_blobs_gradient);
        let forward_time = timeit_loops!(1, {
            // aquire all the locks
            let btm: Vec<_> = self.bottom_blobs_data.iter().map(|b| b.read().unwrap()).collect();
            let wgts: Vec<_> = self.weights_data.iter().map(|w| w.read().unwrap()).collect();
            let tp_ref = self.top_blobs_data.iter().cloned().collect::<Vec<_>>();
            let mut tp = &mut tp_ref.iter().map(|b| b.write().unwrap()).collect::<Vec<_>>();
            let mut top_w = &mut tp.iter_mut().map(|a| a).collect::<Vec<_>>();
            // extract SharedTensors from Blobs
            let weights_data: Vec<&SharedTensor<f32>> = wgts.iter().enumerate().map(|(_, val)| &**val).collect();
            let input_data: Vec<&SharedTensor<f32>> = btm.iter().enumerate().map(|(_, val)| &**val).collect();
            let mut output_data: Vec<&mut SharedTensor<f32>> = top_w.iter_mut().enumerate().map(|(_, val)| &mut ***val).collect();
            self.worker.forward(&self.backend, &input_data, &weights_data, &mut output_data);
        });
        debug!("{:<15} - Forward time: {:.5} ms", &self.name, forward_time / 0.001);
        self.worker.calculate_loss(&self.backend, &mut self.weights_data, &mut self.top_blobs_data)
    }

    /// Uses the underlying layer implementation to compute a backward step.
    ///
    /// See [ILayer.backward](./trait.ILayer.html#method.backward)
    pub fn backward(&mut self) {
        if self.needs_backward {
            debug!("LAYER: {:?}", &self.name);
            self.worker.sync(&self.backend,
                             &mut self.bottom_blobs_data, &mut self.bottom_blobs_gradient,
                             &mut self.weights_data, &mut self.weights_gradient,
                             &mut self.top_blobs_data, &mut self.top_blobs_gradient);
            let top_data: Vec<_> = self.top_blobs_data.iter().map(|b| b.read().unwrap()).collect();
            let top_blobs_data: Vec<&SharedTensor<f32>> = top_data.iter().enumerate().map(|(_, val)| &**val).collect();
            let top_gradient: Vec<_> = self.top_blobs_gradient.iter().map(|b| b.read().unwrap()).collect();
            let top_blobs_gradient: Vec<&SharedTensor<f32>> = top_gradient.iter().enumerate().map(|(_, val)| &**val).collect();
            let wgts_data: Vec<_> = self.weights_data.iter().map(|b| b.read().unwrap()).collect();
            let weights_data: Vec<&SharedTensor<f32>> = wgts_data.iter().enumerate().map(|(_, val)| &**val).collect();
            let bottom_data: Vec<_> = self.bottom_blobs_data.iter().map(|b| b.read().unwrap()).collect();
            let bottom_blobs_data: Vec<&SharedTensor<f32>> = bottom_data.iter().enumerate().map(|(_, val)| &**val).collect();
            let btm_gradient_ref = self.bottom_blobs_gradient.iter().cloned().collect::<Vec<_>>();
            let mut btm_gradient = &mut btm_gradient_ref.iter().map(|b| b.write().unwrap()).collect::<Vec<_>>();
            let mut bottom_gradient = &mut btm_gradient.iter_mut().map(|a| a).collect::<Vec<_>>();
            let mut bottom_blobs_gradient: Vec<&mut SharedTensor<f32>> = bottom_gradient.iter_mut().enumerate().map(|(_, val)| &mut ***val).collect();
            let wgt_gradient_ref = self.weights_gradient.iter().cloned().collect::<Vec<_>>();
            let mut wgt_gradient = &mut wgt_gradient_ref.iter().map(|b| b.write().unwrap()).collect::<Vec<_>>();
            let mut weights_gradient = &mut wgt_gradient.iter_mut().map(|a| a).collect::<Vec<_>>();
            let mut weights_blobs_gradient: Vec<&mut SharedTensor<f32>> = weights_gradient.iter_mut().enumerate().map(|(_, val)| &mut ***val).collect();
            self.worker.backward(&self.backend,
                                 &top_blobs_data,
                                 &top_blobs_gradient,
                                 &weights_data,
                                 &mut weights_blobs_gradient,
                                 &bottom_blobs_data,
                                 &mut bottom_blobs_gradient)
        }
    }

    /// Backpropagation w.r.t. input TODO: DOCS
    pub fn backward_input(&mut self) {
        self.worker.sync(&self.backend,
                         &mut self.bottom_blobs_data, &mut self.bottom_blobs_gradient,
                         &mut self.weights_data, &mut self.weights_gradient,
                         &mut self.top_blobs_data, &mut self.top_blobs_gradient);
        let top_data: Vec<_> = self.top_blobs_data.iter().map(|b| b.read().unwrap()).collect();
        let top_blobs_data: Vec<&SharedTensor<f32>> = top_data.iter().enumerate().map(|(_, val)| &**val).collect();
        let top_gradient: Vec<_> = self.top_blobs_gradient.iter().map(|b| b.read().unwrap()).collect();
        let top_blobs_gradient: Vec<&SharedTensor<f32>> = top_gradient.iter().enumerate().map(|(_, val)| &**val).collect();
        let wgts_data: Vec<_> = self.weights_data.iter().map(|b| b.read().unwrap()).collect();
        let weights_data: Vec<&SharedTensor<f32>> = wgts_data.iter().enumerate().map(|(_, val)| &**val).collect();
        let bottom_data: Vec<_> = self.bottom_blobs_data.iter().map(|b| b.read().unwrap()).collect();
        let bottom_blobs_data: Vec<&SharedTensor<f32>> = bottom_data.iter().enumerate().map(|(_, val)| &**val).collect();
        let btm_gradient_ref = self.bottom_blobs_gradient.iter().cloned().collect::<Vec<_>>();
        let mut btm_gradient = &mut btm_gradient_ref.iter().map(|b| b.write().unwrap()).collect::<Vec<_>>();
        let mut bottom_gradient = &mut btm_gradient.iter_mut().map(|a| a).collect::<Vec<_>>();
        let mut bottom_blobs_gradient: Vec<&mut SharedTensor<f32>> = bottom_gradient.iter_mut().enumerate().map(|(_, val)| &mut ***val).collect();
        self.worker.compute_input_gradient(&self.backend,
                             &weights_data,
                             &top_blobs_data,
                             &top_blobs_gradient,
                             &bottom_blobs_data,
                             &mut bottom_blobs_gradient)
    }

    /// Backpropagation w.r.t. parameters TODO: DOCS
    pub fn backward_parameters(&mut self) {
        self.worker.sync(&self.backend,
                         &mut self.bottom_blobs_data, &mut self.bottom_blobs_gradient,
                         &mut self.weights_data, &mut self.weights_gradient,
                         &mut self.top_blobs_data, &mut self.top_blobs_gradient);
        let top_data: Vec<_> = self.top_blobs_data.iter().map(|b| b.read().unwrap()).collect();
        let top_blobs_data: Vec<&SharedTensor<f32>> = top_data.iter().enumerate().map(|(_, val)| &**val).collect();
        let top_gradient: Vec<_> = self.top_blobs_gradient.iter().map(|b| b.read().unwrap()).collect();
        let top_blobs_gradient: Vec<&SharedTensor<f32>> = top_gradient.iter().enumerate().map(|(_, val)| &**val).collect();
        let wgts_data: Vec<_> = self.weights_data.iter().map(|b| b.read().unwrap()).collect();
        let weights_data: Vec<&SharedTensor<f32>> = wgts_data.iter().enumerate().map(|(_, val)| &**val).collect();
        let bottom_data: Vec<_> = self.bottom_blobs_data.iter().map(|b| b.read().unwrap()).collect();
        let bottom_blobs_data: Vec<&SharedTensor<f32>> = bottom_data.iter().enumerate().map(|(_, val)| &**val).collect();
        let wgt_gradient_ref = self.weights_gradient.iter().cloned().collect::<Vec<_>>();
        let mut wgt_gradient = &mut wgt_gradient_ref.iter().map(|b| b.write().unwrap()).collect::<Vec<_>>();
        let mut weights_gradient = &mut wgt_gradient.iter_mut().map(|a| a).collect::<Vec<_>>();
        let mut weights_blobs_gradient: Vec<&mut SharedTensor<f32>> = weights_gradient.iter_mut().enumerate().map(|(_, val)| &mut ***val).collect();
        self.worker.compute_parameters_gradient(&self.backend,
                             &top_blobs_data,
                             &top_blobs_gradient,
                             &bottom_blobs_data,
                             &mut weights_blobs_gradient)
    }

    /// Synchronize the layers backend.
    pub fn synchronize(&self) {
        self.backend.synchronize().unwrap();
    }

    /// Sets whether the layer should compute gradients w.r.t. a
    /// weight at a particular index given by `weight_id`.
    ///
    /// See [`weight_propagate_down`][1]
    /// ./struct.Layer.html
    pub fn set_weight_propagate_down(&mut self, weight_id: usize, value: bool) {
        if self.weight_propagate_down.len() <= weight_id {
            self.weight_propagate_down.resize(weight_id + 1, true);
        }
        self.weight_propagate_down[weight_id] = value;

    }

    /// TODO
    pub fn bottom_blob_names(&self) -> &[String] {
        &self.bottom_blob_names
    }

    /// Returns the [loss weight][1] associated with the weight blob
    /// with id `weight_id`.
    /// [1]: http://caffe.berkeleyvision.org/tutorial/loss.html
    pub fn loss(&self, weight_id: usize) -> Option<&f32> {
        self.loss.get(weight_id)
    }
}

/// A Layer in a [Neural Network][1] that can handle forward and backward of a computation step.
/// [1]: ../network/index.html
pub trait ILayer<B: IBackend> : ComputeOutput<f32, B> + ComputeInputGradient<f32, B> + ComputeParametersGradient<f32, B> {
    /// Initialize the layer for computation.
    ///
    /// Allows for layer-specific one time setup, e.g. precomputing constant values.
    ///
    /// Is called during [Network][1] initalization.
    /// [1]: ../network/type.Network.html
    fn init(&mut self, backend: Rc<B>) {}

    /// Adjust to shapes of the top blobs to fit the shapes of the bottom blobs.
    ///
    /// Is called during [Network][1] initalization, after [init][2].
    /// [1]: ../network/type.Network.html
    /// [2]: #method.init
    fn reshape(&mut self,
               backend: Rc<B>,
               bottom_data: &[ArcLock<SharedTensor<f32>>],
               weights_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               top_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               top_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>) {}

    /// Compute the [feedforward][1] layer output using the provided Backend.
    /// [1]: https://en.wikipedia.org/wiki/Feedforward_neural_network
    ///
    /// Aquires read locks for the bottom blobs ([ReadBlob][2])
    /// and write locks for the top blobs ([WriteBlob][3]) to ensure sequential computation,
    /// and then passes them to computation method specific function ([forward_cpu][4]).
    ///
    /// [2]: ./type.ReadBlob.html
    /// [3]: ./type.WriteBlob.html
    /// [3]: #method.forward_cpu
    #[cfg_attr(lint, allow(map_clone))]
    fn forward(&self,
               backend: &B,
               input_data: &[&SharedTensor<f32>],
               weights_data: &[&SharedTensor<f32>],
               output_data: &mut [&mut SharedTensor<f32>]
           ) {
        self.compute_output(backend, weights_data, input_data, output_data);
    }

    /// Calculate the loss for the top blobs in the layer.
    ///
    /// If `loss_weight(i)` returns `NAN` for a blob, no loss will be calculated for that blob.
    ///
    /// `calculate_loss` is called at the end of the forward computation step.
    fn calculate_loss(&self, backend: &B, weights: &mut Vec<ArcLock<SharedTensor<f32>>>, top: &mut Vec<ArcLock<SharedTensor<f32>>>) -> f32 {
        let mut loss = 0f32;

        let tp_ref = top.iter().cloned().collect::<Vec<_>>();
        let tp = &mut tp_ref.iter().map(|b| b.write().unwrap()).collect::<Vec<_>>();
        let wgts_ref = weights.iter().cloned().collect::<Vec<_>>();
        let wgts = &mut wgts_ref.iter().map(|b| b.write().unwrap()).collect::<Vec<_>>();

        for (top_id, (top_layer, weight)) in tp.iter_mut().zip(wgts).enumerate() {
            if self.loss_weight(top_id).is_nan() {
                debug!("Top Id {:?} does not contribute to loss - Skipping loss calculation.", &top_id);
                continue;
            }

            let top_blob = top_layer;
            let data = top_blob;
            let loss_weights = weight;

            let native_backend = native_backend();
            match data.add_device(native_backend.device()) { _ => data.sync(native_backend.device()).unwrap() }
            match loss_weights.add_device(native_backend.device()) { _ => loss_weights.sync(native_backend.device()).unwrap() }
            let mut shared_loss = SharedTensor::<f32>::new(native_backend.device(), &vec![1]).unwrap();
            native_backend.dot_plain(data, loss_weights, &mut shared_loss).unwrap();
            let native_loss = shared_loss.get(native_backend.device()).unwrap().as_native().unwrap();
            loss += native_loss.as_slice::<f32>()[0];
        }

        loss
    }

    /// Compute the [backpropagation][1] layer output and gradient using the currently set computation method.
    /// [1]: https://en.wikipedia.org/wiki/Backpropagation
    ///
    /// Aquires read locks for the top blobs ([ReadBlob][2])
    /// and write locks for the bottom blobs ([WriteBlob][3]) to ensure sequential computation,
    /// and then passes them to computation method specific function ([backward_cpu][4]).
    ///
    /// [2]: ./type.ReadBlob.html
    /// [3]: ./type.WriteBlob.html
    /// [3]: #method.backward_cpu
    #[cfg_attr(lint, allow(map_clone))]
    fn backward(&self,
                backend: &B,
                top_data: &[&SharedTensor<f32>],
                top_gradients: &[&SharedTensor<f32>],
                weights_data: &[&SharedTensor<f32>],
                weights_gradients: &mut [&mut SharedTensor<f32>],
                bottom_data: &[&SharedTensor<f32>],
                bottom_gradients: &mut [&mut SharedTensor<f32>]) {
        self.compute_input_gradient(backend, weights_data, top_data, top_gradients, bottom_data, bottom_gradients);
        self.compute_parameters_gradient(backend, top_data, top_gradients, bottom_data, weights_gradients);
    }

    /// Synchronize the blobs before doing a forward or backward operation.
    ///
    /// This is necessary because the forward_layer and backward_layer methods only immutably
    /// borrow the corresponding input blobs and weights which they are not supposed to change.
    /// However synchronizing all blobs to the same device may be neccessary for some computations,
    /// which can only be done with a mutable borrow.
    fn sync(&self,
            backend: &B,
            input_data: &mut [ArcLock<SharedTensor<f32>>],
            input_gradients: &mut [ArcLock<SharedTensor<f32>>],
            weights_data: &mut [ArcLock<SharedTensor<f32>>],
            weights_gradients: &mut [ArcLock<SharedTensor<f32>>],
            output_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
            output_gradients: &mut Vec<ArcLock<SharedTensor<f32>>>) {
        if self.sync_native() {
            let backend = native_backend();
            for tensor in input_data {
                let mut sync = tensor.write().unwrap();
                match sync.add_device(backend.device()) { _ => sync.sync(backend.device()).unwrap() }
            }
            for tensor in input_gradients {
                let mut sync = tensor.write().unwrap();
                match sync.add_device(backend.device()) { _ => sync.sync(backend.device()).unwrap() }
            }
            for tensor in weights_data {
                let mut sync = tensor.write().unwrap();
                match sync.add_device(backend.device()) { _ => sync.sync(backend.device()).unwrap() }
            }
            for tensor in weights_gradients {
                let mut sync = tensor.write().unwrap();
                match sync.add_device(backend.device()) { _ => sync.sync(backend.device()).unwrap() }
            }
            for tensor in output_data {
                let mut sync = tensor.write().unwrap();
                match sync.add_device(backend.device()) { _ => sync.sync(backend.device()).unwrap() }
            }
            for tensor in output_gradients {
                let mut sync = tensor.write().unwrap();
                match sync.add_device(backend.device()) { _ => sync.sync(backend.device()).unwrap() }
            }
        } else {
            for tensor in input_data {
                let mut sync = tensor.write().unwrap();
                match sync.add_device(backend.device()) { _ => sync.sync(backend.device()).unwrap() }
            }
            for tensor in input_gradients {
                let mut sync = tensor.write().unwrap();
                match sync.add_device(backend.device()) { _ => sync.sync(backend.device()).unwrap() }
            }
            for tensor in weights_data {
                let mut sync = tensor.write().unwrap();
                match sync.add_device(backend.device()) { _ => sync.sync(backend.device()).unwrap() }
            }
            for tensor in weights_gradients {
                let mut sync = tensor.write().unwrap();
                match sync.add_device(backend.device()) { _ => sync.sync(backend.device()).unwrap() }
            }
            for tensor in output_data {
                let mut sync = tensor.write().unwrap();
                match sync.add_device(backend.device()) { _ => sync.sync(backend.device()).unwrap() }
            }
            for tensor in output_gradients {
                let mut sync = tensor.write().unwrap();
                match sync.add_device(backend.device()) { _ => sync.sync(backend.device()).unwrap() }
            }
        }
    }

    /// Return whether "anonymous" top blobs are created automatically for the layer.
    ///
    /// If this method returns true, Network::init will create enough "anonymous" top
    /// blobs to fulfill the requirement specified by [exact_num_top_blobs][1] or
    /// [min_top_blobs][2].
    /// [1]: #method.exact_num_top_blobs
    /// [2]: #method.min_top_blobs
    fn auto_top_blobs(&self) -> bool {
        false
    }
    /// Returns the minimum number of top blobs required by the layer,
    /// or 0 if no minimum number is required.
    ///
    /// This method should be overridden to return a positive value if your
    /// layer expects some minimum number of top blobs.
    fn min_top_blobs(&self) -> usize {
        0
    }
    /// Returns the exact number of top blobs required by the layer,
    /// or 0 if no exact number is required.
    ///
    /// This method should be overridden to return a positive value if your
    /// layer expects some exact number of top blobs.
    fn exact_num_top_blobs(&self) -> usize {
        0
    }
    /// Returns the exact number of bottom blobs required by the layer,
    /// or 0 if no exact number is required.
    ///
    /// This method should be overridden to return a positive value if your
    /// layer expects some exact number of bottom blobs.
    fn exact_num_bottom_blobs(&self) -> usize {
        0
    }
    /// Return whether to allow force_backward for a given bottom blob index.
    ///
    /// If allow_force_backward(i) == false, we will ignore the force_backward
    /// setting and backpropagate to blob i only if it needs gradient information
    /// (as is done when force_backward == false).
    fn allow_force_backward(&self, bottom_id: usize) -> bool {
        true
    }
    /// Return wether a simple native backend should be used to [sync][1] instead of the default backend.
    /// [1]: #method.sync
    ///
    /// If `false` is returned the default backend will be used, otherwise a new native backend
    /// will be created and provided as argument to `sync`.
    fn sync_native(&self) -> bool {
        false
    }

    /// Return the associated loss weight for a given top blob index.
    ///
    /// If loss_weight(i) == NAN, no loss will be calculated for the top blob.
    ///
    /// This is usually overridden by loss layers.
    fn loss_weight(&self, top_id: usize) -> f32 {
        ::std::f32::NAN
    }
}

/// A Layer that can compute the output (= top) for a given input (= bottom).
pub trait ComputeOutput<T, B: IBackend> {
    /// Compute output for given input and write them into `output_data`.
    fn compute_output(&self,
                      backend: &B,
                      weights_data: &[&SharedTensor<T>],
                      input_data: &[&SharedTensor<T>],
                      output_data: &mut [&mut SharedTensor<T>]);
}

/// A Layer that can compute the gradient with respect to its input (= bottom).
pub trait ComputeInputGradient<T, B: IBackend> {
    /// Compute gradients with respect to the inputs and write them into `input_gradients`.
    fn compute_input_gradient(&self,
                              backend: &B,
                              weights: &[&SharedTensor<T>],
                              output_data: &[&SharedTensor<T>],
                              output_gradients: &[&SharedTensor<T>],
                              input_data: &[&SharedTensor<T>],
                              input_gradients: &mut [&mut SharedTensor<T>]);
}

/// A Layer that can compute the gradient with respect to its parameters (= weights, bias, etc.).
pub trait ComputeParametersGradient<T, B: IBackend> {
    /// Compute gradients with respect to the parameters and write them into `parameters_gradients`.
    fn compute_parameters_gradient(&self,
                                   backend: &B,
                                   output_data: &[&SharedTensor<T>],
                                   output_gradients: &[&SharedTensor<T>],
                                   input_data: &[&SharedTensor<T>],
                                   parameters_gradients: &mut [&mut SharedTensor<T>]) {}
}

impl<B: IBackend> fmt::Debug for ILayer<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", "foo", "bar")
    }
}

#[derive(Debug, Clone)]
/// Layer Configuration Struct
pub struct LayerConfig {
    /// The name of the Layer
    pub name: String,

    /// The type of the Layer
    pub layer_type: LayerType,

    /// The name for each top Blob
    pub tops: Vec<String>,

    /// The name for each bottom Blob
    pub bottoms: Vec<String>,

    /// Specifies training configuration for each weight blob.
    pub params: Vec<WeightConfig>,

    /// Specifies on which bottoms the backpropagation should be skipped.
    /// The size must be either 0 or equal to the number of bottoms.
    pub propagate_down: Vec<bool>,
}

#[derive(Debug, Clone)]
/// The Layer Types
pub enum LayerType {
    // Common layers
    /// Convolution Layer
    Convolution(ConvolutionConfig),
    /// FullyConnected Layer
    FullyConnected(FullyConnectedConfig),
    /// Pooling Layer
    Pooling(PoolingConfig),
    // Activation layers
    /// ReLU Layer
    ReLU,
    /// Sigmoid Layer
    Sigmoid,
    // Loss layers
    /// SoftmaxLoss Layer
    SoftmaxLoss,
}

impl LayerConfig {
    /// Creates a new LayerConfig
    pub fn new(name: String, layer_type: LayerType) -> LayerConfig {
        LayerConfig {
            name: name,
            layer_type: layer_type,

            tops: Vec::new(),
            bottoms: Vec::new(),

            params: Vec::new(),
            propagate_down: Vec::new(),
        }
    }

    /// Returns the Name of the requested top Blob
    pub fn top(&self, top_id: usize) -> Option<&String> {
        self.tops.get(top_id)
    }

    /// Returns the number of top Blobs
    pub fn tops_len(&self) -> usize {
        self.tops.len()
    }

    /// Returns the Name of the requested bottom Blob
    pub fn bottom(&self, bottom_id: usize) -> Option<&String> {
        self.bottoms.get(bottom_id)
    }

    /// Returns the number of bottom Blobs
    pub fn bottoms_len(&self) -> usize {
        self.bottoms.len()
    }

    /// Returns the requested WeightConfig
    pub fn param(&self, param_id: usize) -> Option<&WeightConfig> {
        self.params.get(param_id)
    }

    /// Returns the number of params
    pub fn params_len(&self) -> usize {
        self.params.len()
    }

    /// Check if the configured parameters make sense.
    pub fn validate(&self) -> Result<(), &'static str> {
        try!(self.validate_propagate_down_len());
        Ok(())
    }

    /// Checks if propagate down length makes sense.
    fn validate_propagate_down_len(&self) -> Result<(), &'static str> {
        if self.propagate_down.is_empty() || self.propagate_down.len() == self.bottoms.len() {
            Ok(())
        } else {
            Err("propagate_down config must be specified either 0 or bottom_size times")
        }
    }

    // /// Checks if propagate down length is sane
    // pub fn check_propagate_down_len(&self) -> bool {
    //     self.propagate_down.is_empty() || self.propagate_down.len() == self.bottoms.len()
    // }
}
