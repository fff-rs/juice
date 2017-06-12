extern crate coaster;
use coaster::prelude::*;

fn main() {
    let ntv = Native::new();
    let dev = ntv.new_device(ntv.hardwares()).unwrap();

    let mem = {
        let x = &mut SharedTensor::<f32>::new(&10);
        //~^ ERROR error: borrowed value does not live long enough
        x.write_only(&dev).unwrap();
        let m = x.read(&dev).unwrap();
        m
    };
}

