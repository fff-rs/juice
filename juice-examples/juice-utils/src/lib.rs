extern crate bytes;
extern crate flate2;

use flate2::read::GzDecoder;
use reqwest;
use reqwest::blocking::Client;
use std::fs::File;
use std::io::prelude::*;


pub fn download_datasets(datasets: &[&str], asset_path: &str, base_url: &str) -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();
    std::fs::create_dir_all(asset_path)?;
    for dataset in datasets {
        let url = format!("{}/{}", base_url, dataset);
        let resp = client.get(&url).send()?.bytes()?;
        println!("Downloading {}", dataset);
        let name = format!("{}/{}", asset_path, dataset);
        let mut f = File::create(name.clone()).expect("Failed to create file");
        f.write(&resp).unwrap();
    }
    Ok(())
}

pub fn unzip_datasets(datasets: &[&str], asset_path: &str) {
    for filename in datasets {
        println!("Decompressing: {}", filename);

        let mut file_handle = File::open(&format!("{}/{}", asset_path, filename)).unwrap();
        let mut in_file: Vec<u8> = Vec::new();
        let mut decompressed_file: Vec<u8> = Vec::new();

        file_handle.read_to_end(&mut in_file).unwrap();

        let mut decoder = GzDecoder::new(in_file.as_slice());

        decoder.read_to_end(&mut decompressed_file).unwrap();

        let filename_string = filename.split(".").nth(0).unwrap();

        File::create(format!("{}/{}", asset_path, filename_string))
            .unwrap()
            .write_all(&decompressed_file as &[u8])
            .unwrap();
    }
}