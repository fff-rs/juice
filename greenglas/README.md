# Greenglas • [![Join the chat at https://gitter.im/spearow/greenglas](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/spearow/greenglas?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Greenglas Build Status](https://ci.spearow.io/api/v1/pipelines/leaf/jobs/test-greenglas/badge)](https://ci.spearow.io/teams/main/pipelines/leaf) [![Crates.io](https://img.shields.io/crates/v/greenglas.svg)](https://crates.io/crates/greenglas) [![License](https://img.shields.io/crates/l/greenglas.svg)](#license)

Greenglas tries to provide a smart and customizable pipeline for preprocessing
data for machine learning tasks. Clean preprocessing methods for the most
common type of data, makes preprocessing easy. Greenglas offers a pipeline of
Modifiers and Transformers to turn non-numeric data into a safe and consistent
numeric output in the form of Coaster's [`SharedTensor`](https://github.com/spearow/coaster). For
putting your preprocessed data to use, you might like to use the Machine
Learning Framework [`Leaf`](https://github.com/spearow/leaf).

For more information see the [Documentation](http://spearow.github.io/greenglas).

## Architecture

Greenglas exposes several standard data types, which might need a numeric
transformation in order to be processed by a Machine Learning Algorithm such as
Neural Nets.

Data Types can be modified through Modifiers. This provides a coherent interface,
allowing for custom modifiers. You can read more about custom modifiers further
down. First, an example of a Data Type modification:

```
let mut data_type = Image { value: ... }
data_type = data_type.set((ModifierOne(param1, param2), ModifierTwo(anotherParam));
image.set(Resize(20, 20))
```

After one, none or many modifications through Modifiers, the Data Type can then
finally be transformed into a [`SharedTensor`](https://github.com/spearow/coaster)
(numeric Vector). Taking `data_type` from the above example:

```
// the Vector secures the correct shape and capacity of the final SharedTensor
let final_tensor = data_type.transform(vec![20, 20, 3]).unwrap();
```

## Transformable Data Types

These are the data types that `greenglas` is currently addressing. For most of
them are basic Modifiers and Transformers already specified.

- **`Missing`**: `NULL` data
- **`Label`**: labeled data such as ID's, Categories, etc.
- **`Word`**: a String of arbitrary lengths
- **`Image`**
- **`Audio`**

## Modifiers

All Modifiers implement the `Modifier` trait from
[`rust-modifier`](https://github.com/reem/rust-modifier). As all Transformable
Data Types implement the `Set` trait of the same library, one can easily write
custom modifiers as well. Quick Example:

```
extern crate greenglas;

use greenglas::Image;
use greenglas::modifier::Modifier;

struct CustomModifier(usize)

impl Modifier<Image> for CustomModifier {
    fn modify(self, image: &mut Image) {
        image.value = some_extern_image_manipulation_fn(self.0);
    }
}
```

## Contributing

Want to contribute? Awesome! We have [instructions to help you get started contributing code or documentation](CONTRIBUTING.md).

Autumn has a mostly real-time collaboration culture and happens on the [Spearow
Gitter Channels](https://gitter.im/spearow). Or you reach out to the
Maintainer(s). e.g.
{[@drahnr](https://github.com/drahnr), }.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as below, without any additional terms or
conditions.

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
