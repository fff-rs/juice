# cuticula
[![Join the chat at https://gitter.im/storeness/cuticula](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/storeness/cuticula?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/storeness/cuticula.svg?branch=master)](https://travis-ci.org/storeness/cuticula) [![Coverage Status](https://coveralls.io/repos/storeness/cuticula/badge.svg?branch=master&service=github)](https://coveralls.io/github/storeness/cuticula?branch=master)

Cuticula provides a convenient and customizable interface for modifying and
transforming (non-numeric) data into numeric data, that can be used for machine
learning purposes.

See the [Documentation](http://storeness.github.io/cuticula).

## Architecture

Cuticula exposes several standard data types, which might need a numeric
transformation in order to be processed by a Machine Learning Algorithm such as
Neural Nets.

Data Types can be modified through Modifiers. This provides a coherent interface,
allowing for custom modifiers. You can read more about custom modifiers further
down. First, an example of a Data Type modification:

```
let mut data_type = Image { value: ... }
data_type = data_type.set((ModifierOne(param1, param2), ModifierTwo(anotherParam));
```

After one, none or many modifications through Modifiers, the Data Type can then
finally be transformed into a numeric Vector/Matrix. Taking `data_type` from the above example:

```
// 3 specifies the dimensions of the numeric output
data_type = data_type.transform(3);
```

## Transformable Data Types

These are the data types that `cuticula` is currently addressing. For most of
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
extern crate cuticula;

use cuticula::Image;
use cuticula::modifier::Modifier;

struct CustomModifier(usize)

impl Modifier<Image> for CustomModifier {
    fn modify(self, image: &mut Image) {
        image.value = some_extern_image_manipulation_fn(self.0);
    }
}
```
