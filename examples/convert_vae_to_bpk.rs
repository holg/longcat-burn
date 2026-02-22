//! Convert WAN VAE weights from safetensors to Burn's bpk format
//!
//! Usage:
//!   cargo run --example convert_vae_to_bpk --release --features metal -- \
//!     --input /path/to/wan_vae.safetensors \
//!     --output /path/to/wan_vae.bpk

use std::path::PathBuf;

use burn::backend::candle::{Candle, CandleDevice};
use burn::store::{BurnpackStore, ModuleStore};
use half::bf16;
use longcat_burn::modules::vae::{WanVae, WanVaeConfig};

// Use Candle with BF16 to match the model weights
type Backend = Candle<bf16, i64>;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();

    // Default paths
    let mut input_path = PathBuf::from("/Volumes/tb3ssd/volume/burn_models/wan_vae.safetensors");
    let mut output_path = PathBuf::from("/Volumes/tb3ssd/volume/burn_models/wan_vae.bpk");

    // Parse arguments
    if let Some(i) = args.iter().position(|x| x == "--input" || x == "-i") {
        if i + 1 < args.len() {
            input_path = PathBuf::from(&args[i + 1]);
        }
    }

    if let Some(i) = args.iter().position(|x| x == "--output" || x == "-o") {
        if i + 1 < args.len() {
            output_path = PathBuf::from(&args[i + 1]);
        }
    }

    if args.iter().any(|x| x == "--help" || x == "-h") {
        eprintln!("Convert WAN VAE weights from safetensors to bpk format");
        eprintln!();
        eprintln!("Usage: convert_vae_to_bpk [OPTIONS]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  -i, --input <PATH>   Input safetensors file");
        eprintln!("  -o, --output <PATH>  Output bpk file");
        eprintln!("  -h, --help           Show this help");
        return Ok(());
    }

    println!("WAN VAE Converter - safetensors to bpk");
    println!("======================================");
    println!();
    println!("Input: {:?}", input_path);
    println!("Output: {:?}", output_path);

    let config = WanVaeConfig::default();
    println!("\nConfig:");
    println!("  Latent channels: {}", config.latent_channels);
    println!("  Compression: 4x8x8 (temporal x spatial)");

    println!("\nInitializing model on CPU...");
    let device = CandleDevice::Cpu;
    let mut vae: WanVae<Backend> = config.init(&device);

    println!("Loading weights from safetensors...");
    vae.load_weights(&input_path)
        .map_err(|e| format!("Failed to load weights: {e:?}"))?;

    println!("Saving to bpk format...");
    let mut store = BurnpackStore::from_file(&output_path)
        .auto_extension(false);
    store.collect_from(&vae)
        .map_err(|e| format!("Failed to save: {e:?}"))?;

    // Print file sizes
    if let Ok(meta) = std::fs::metadata(&input_path) {
        println!("\nInput size: {:.2} MB", meta.len() as f64 / 1e6);
    }
    if let Ok(meta) = std::fs::metadata(&output_path) {
        println!("Output size: {:.2} MB", meta.len() as f64 / 1e6);
    }

    println!("\nConversion complete!");

    Ok(())
}
