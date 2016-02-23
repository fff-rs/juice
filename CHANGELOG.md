<a name="0.3.1"></a>
## 0.3.1 (2016-02-23)


#### Features

* **pointwise:**  add pointwise activation functions (cuDNN) ([d74821b5](https://github.com/autumnai/collenchyma-nn/commit/d74821b582056f9acd3bdb4acd98f72668d070f8))



<a name="0.3.0"></a>
## 0.3.0 (2016-02-22)


#### Features

* **log_softmax:**  add LogSoftmax operations ([86a8ae67](https://github.com/autumnai/collenchyma-nn/commit/86a8ae67727e0a5d28c901a7a32940fd7e2250f2))
* **cuda:**
  * share workspace between CUDA convolution operations ([7f5f3207](https://github.com/autumnai/collenchyma-nn/commit/7f5f3207873874accb7a5a16d637e2701161ac04))
  * allow CUDA activations to work with 1D/2D tensors ([f4effe7d](https://github.com/autumnai/collenchyma-nn/commit/f4effe7d66d96537251d86bf24968b521a951121))
  * allow CUDA softmax to work with 1-3D tensors ([f74f72b6](https://github.com/autumnai/collenchyma-nn/commit/f74f72b6207505f4c29c7c44a9748d83972e7f72))
* **nn_trait:**  remove trait bounds for NN ([9ad08d9f](https://github.com/autumnai/collenchyma-nn/commit/9ad08d9f97cc382699c78c1397b52509d2e98969))
* **license:**  change license to dual MIT/Apache-2.0 ([8a940690](https://github.com/autumnai/collenchyma-nn/commit/8a940690e21bae269c44b9501e956bbf066cdcc1))

#### Breaking Changes

* **convolution:**  implement convolutions correctly ([24b164b5](https://github.com/autumnai/collenchyma-nn/commit/24b164b55a913f522d79832308cf2e4a7996612a))

#### Performance

* **convolution:**  don't do a memAlloc for a zero size workspace ([73612bb5](https://github.com/autumnai/collenchyma-nn/commit/73612bb56ab70500b4670b7a9a12390e2facee37))


<a name="0.2.1"></a>
## 0.2.1 (2016-01-21)


#### Features

* **native:**  Add support for softmax w/ test and benches. ([14d6d1bc](https://github.com/autumnai/collenchyma-nn/commit/14d6d1bcda8bbc0ffa368527633f592862517200))

#### Bug Fixes

* **native:**  Fix sigmoid_grad to use x_diff instead of x for dx ([c25a32aa](https://github.com/autumnai/collenchyma-nn/commit/c25a32aa272ff3c753ee8be2ea89457367b38734))



<a name="0.2.0"></a>
## 0.2.0 (2016-01-15)


#### Features

* **bench:**  add bench and perf utilities ([0e2d34c6](https://github.com/autumnai/collenchyma-nn/commit/0e2d34c67acba38c6910cdff6e983b5285dfb852))
* **native:**  implement Sigmoid, ReLU, tanh for Native backend. ([ece54e37](https://github.com/autumnai/collenchyma-nn/commit/ece54e37a241f81b45888225ab0ee28c538950f6))


<a name="0.1.0"></a>
## 0.1.0 (2015-12-21)


#### Bug Fixes

* **scale_params:**  fix ScalParams default to work on stable ([43654dca](https://github.com/autumnai/collenchyma-nn/commit/43654dca7cb92826ffecd4f0cd251fb7071d11c5))

#### Features

* **activation:**  add most popular NN activation functions ([3311bb43](https://github.com/autumnai/collenchyma-nn/commit/3311bb43d78c850db8322c9ea8c1a5f2ca189cd1))
* **features:**  add framework feature groups ([08629ea8](https://github.com/autumnai/collenchyma-nn/commit/08629ea8f1c38047a5d7fec24601e21ba79d704f))
* **nn:**
  *  add all cudnn available operations to collenchyma-nn ([03384763](https://github.com/autumnai/collenchyma-nn/commit/033847630a0674c372666db209d436a80ecabe1b))
  *  add basic nn implementation structure ([aa17ef0f](https://github.com/autumnai/collenchyma-nn/commit/aa17ef0f5064e479152ac3e398bf64887e03b6e2))
* **sigmoid:**
  *  add full sigmoid CUDA implementation ([8ea1a290](https://github.com/autumnai/collenchyma-nn/commit/8ea1a29016c364536755e2fb5d13a52352b059ab))
  *  add CUDA Sigmoid ([6aceb957](https://github.com/autumnai/collenchyma-nn/commit/6aceb957d05a0ee625b48bab38693b99c9e09f01))
