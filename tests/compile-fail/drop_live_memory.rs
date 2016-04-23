extern crate collenchyma;
use collenchyma::prelude::*;

fn main() {
    let ntv = Native::new();
    let dev = ntv.new_device(ntv.hardwares()).unwrap();

    let x = &mut SharedTensor::<f32>::new(&10);
    let m = x.write_only(&dev).unwrap();
    x.drop_device(&dev);
    //~^ ERROR error: cannot borrow `*x` as mutable more than once at a time
}

