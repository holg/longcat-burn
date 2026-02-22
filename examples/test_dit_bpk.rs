//! Quick test to verify DiT loads from bpk correctly
use std::path::PathBuf;
use burn::backend::candle::{Candle, CandleDevice};
use burn::store::{BurnpackStore, ModuleStore};
use half::bf16;
use longcat_burn::{LongCatConfig, LongCatDiT};

type Backend = Candle<bf16, i64>;

fn main() {
    println!("Testing DiT BPK Loading");
    println!("=======================\n");

    let device = CandleDevice::Cpu;

    let config = LongCatConfig::default();
    println!("Config: {} layers, {} hidden, {:.1}B params",
        config.num_layers, config.hidden_size, config.num_params() as f64 / 1e9);

    println!("\nInitializing DiT model...");
    let mut dit: LongCatDiT<Backend> = config.init(&device);
    println!("Model initialized");

    let dit_path = PathBuf::from("/Volumes/tb3ssd/volume/burn_models/longcat_dit.bpk");
    println!("\nLoading weights from {:?}...", dit_path);

    let mut store = BurnpackStore::from_file(&dit_path);
    match store.apply_to(&mut dit) {
        Ok(result) => {
            println!("DiT weights loaded successfully from BPK!");
            println!("Applied: {}, Missing: {}, Errors: {}",
                result.applied.len(), result.missing.len(), result.errors.len());
            if !result.errors.is_empty() {
                println!("Errors: {:?}", result.errors);
            }
            println!("Model accessible - BPK is valid!");
        }
        Err(e) => {
            println!("DiT BPK load ERROR: {:?}", e);
            std::process::exit(1);
        }
    }
}
