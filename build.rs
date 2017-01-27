extern crate capnpc;

fn main() {
    ::capnpc::CompilerCommand::new().src_prefix("capnp").file("capnp/leaf.capnp").run().expect("compiling schema");
}
