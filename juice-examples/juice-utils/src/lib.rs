use flate2::read::GzDecoder;
use fs_err as fs;
use reqwest::blocking::Client;
use std::io;
use std::io::prelude::*;

pub fn download_datasets(
    datasets: &[&str],
    asset_path: &str,
    base_url: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();
    std::fs::create_dir_all(asset_path)?;
    for dataset in datasets {
        let url = format!("{}/{}", base_url, dataset);
        log::info!("Downloading {}", dataset);
        let resp = client.get(&url).send()?.bytes()?;
        let name = format!("{}/{}", asset_path, dataset);
        let mut f = fs::File::create(name.clone()).expect("Failed to create file");
        f.write_all(&resp)?;
        log::info!("Download of {} complete.", dataset);
    }
    Ok(())
}

pub fn unzip_datasets(datasets: &[&str], asset_path: &str) -> io::Result<()> {
    for filename in datasets {
        log::info!("Decompressing: {}", filename);

        let file_handle = fs::File::open(&format!("{}/{}", asset_path, filename)).unwrap();
        let mut decoder = GzDecoder::new(file_handle);

        let filename_string = filename.split(".").nth(0).unwrap();

        let mut dest = fs::File::create(format!("{}/{}", asset_path, filename_string))?;
        std::io::copy(&mut decoder, &mut dest)?;
        log::info!("Decompression of {} complete.", filename);
    }
    Ok(())
}
