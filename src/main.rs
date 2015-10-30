#![feature(plugin)]
#![plugin(docopt_macros)]
extern crate rustc_serialize;
extern crate docopt;
extern crate csv;
extern crate cuticula;
extern crate hyper;

use std::io::prelude::*;
use std::fs::File;

use hyper::Client;

use docopt::Docopt;
use csv::{Reader};
use cuticula::{Transformer, Image};

docopt!(Args derive Debug, "
Leaf Examples

Usage:
    leaf-examples load-dataset <dataset-name>
    leaf-examples mnist
    leaf-examples (-h | --help)
    leaf-examples --version

Options:
    -h --help     Show this screen.
    --version     Show version.
");

#[cfg(not(test))]
#[allow(unused_must_use)]
fn main() {
    // Parse Arguments
    let args: Args = Args::docopt().decode().unwrap_or_else(|e| e.exit());

    if args.cmd_load_dataset {
        match args.arg_dataset_name.as_ref() {
            "mnist" => {
                let datasets = ["mnist_test.csv", "mnist_train.csv"];
                for (i, v) in datasets.iter().enumerate() {
                    println!("Downloading... {}/{}: {}", i+1, datasets.len(), v);
                    let mut body = String::new();

                    Client::new()
                        .get(&format!("http://pjreddie.com/media/files/{}", v))
                        .send().unwrap().read_to_string(&mut body);

                    File::create(format!("assets/{}", v))
                        .unwrap()
                        .write_all(&body.into_bytes());
                }
                println!("{}", "awesome".to_string())
            },
            _ => println!("{}", "fail".to_string())
        }
    } else if args.cmd_mnist {
        let mut rdr = Reader::from_file("assets/mnist_train.csv").unwrap();
        for row in rdr.decode() {
            match row {
                Ok(value) => {
                    let row_vec: Box<Vec<u8>> = Box::new(value);
                    let label = row_vec[0];
                    let pixels = row_vec[1..].to_vec();
                    let img = Image::from_luma_pixels(28, 28, pixels);
                    match img {
                        Ok(i) => {
                            println!("({}): {:?}", label, i.transform(vec![28, 28]));
                        },
                        Err(_) => unimplemented!()
                    }
                },
                _ => println!("{}", "no value".to_string())
            }
        }
    }
}
