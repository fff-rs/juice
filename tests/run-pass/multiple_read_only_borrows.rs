extern crate collenchyma;
use collenchyma::prelude::*;

fn main() {
    let ntv = Native::new();
    let dev = ntv.new_device(ntv.hardwares()).unwrap();

    let x = &mut SharedTensor::<f32>::new(&10).unwrap();
    x.write_only(&dev).unwrap();

    let m1 = x.read(&dev);
    let m2 = x.read(&dev);
    let m3 = x.read(&dev);
}

