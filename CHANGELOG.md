<a name="1.2.0"></a>
## 1.2.0 (2016-02-21)

#### Features

* **log_softmax:**  add logarithmic softmax (`log_softmax`) ([2147ccec](https://github.com/autumnai/rust-cudnn/commit/2147ccec328f79662f9662ce0659f228964c2533))


<a name="1.1.0"></a>
## 1.1.0 (2016-02-02)

#### Breaking Changes

* **convolution:**
  *  make parameter ordering more consistent ([bb314bdd](https://github.com/autumnai/rust-cudnn/commit/bb314bdd1ddd8213539252bb4bc0f5ba514e5888))
  *  implement backward _data _filter _bias ([09cdeb7a](https://github.com/autumnai/rust-cudnn/commit/09cdeb7ac48dc77aae1db30b70579b030349bd4f))

#### Bug Fixes

* **convolution:**  make parameter ordering more consistent ([bb314bdd](https://github.com/autumnai/rust-cudnn/commit/bb314bdd1ddd8213539252bb4bc0f5ba514e5888))

#### Features

* **convolution:**  implement backward _data _filter _bias ([09cdeb7a](https://github.com/autumnai/rust-cudnn/commit/09cdeb7ac48dc77aae1db30b70579b030349bd4f))
* **license:**  change license to dual MIT/Apache-2.0 ([1d37a0fe](https://github.com/autumnai/rust-cudnn/commit/1d37a0fe149f95b2b895876aa811d3dc86a957f9)), closes [#10](https://github.com/autumnai/rust-cudnn/issues/10)

<a name="1.0.1"></a>
## 1.0.1 (2015-12-21)

#### Bug Fixes

* **pooling:**  fix pooling and convolution ([5b8c94b0](https://github.com/autumnai/rust-cudnn/commit/5b8c94b06673ca4f9ef0c218addf774fcab578d7))

<a name="1.0.0"></a>
## 1.0.0 (2015-12-16)

First working release
