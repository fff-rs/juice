extern crate capnpc;

fn main() {
    ::capnpc::CompilerCommand::new().src_prefix("capnp").file("capnp/juice.capnp").run().expect("schema compiler command");
}
