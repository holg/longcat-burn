//! Example: Generate video from text prompt using LongCat
//!
//! This example demonstrates how to use the LongCat video generation pipeline
//! for text-to-video (T2V), image-to-video (I2V), and video continuation.
//!
//! # Usage
//!
//! ```bash
//! # Build with Metal backend (macOS)
//! cargo run --example generate --features metal --release
//!
//! # Build with CUDA backend (Windows/Linux)
//! cargo run --example generate --features cuda --release
//!
//! # Build with CPU backend
//! cargo run --example generate --features cpu --release
//! ```

use std::path::PathBuf;

use longcat_burn::{GenerateConfig, LongCatPipeline, PipelineBuilder};

/// Find weights file, preferring .bpk over .safetensors
fn find_weights(base_path: &str) -> Option<PathBuf> {
    let bpk = PathBuf::from(format!("{}.bpk", base_path));
    if bpk.exists() {
        return Some(bpk);
    }
    let safetensors = PathBuf::from(format!("{}.safetensors", base_path));
    if safetensors.exists() {
        return Some(safetensors);
    }
    None
}

// Backend and device type selection based on features
#[cfg(feature = "metal")]
mod backend {
    pub type Backend = burn::backend::candle::Candle;
    pub type Device = burn::backend::candle::CandleDevice;

    pub fn get_device() -> Device {
        Device::metal(0)
    }
}

#[cfg(all(feature = "cuda", not(feature = "metal")))]
mod backend {
    pub type Backend = burn::backend::candle::Candle;
    pub type Device = burn::backend::candle::CandleDevice;

    pub fn get_device() -> Device {
        Device::cuda(0)
    }
}

#[cfg(all(feature = "cpu", not(feature = "metal"), not(feature = "cuda")))]
mod backend {
    pub type Backend = burn::backend::ndarray::NdArray<f32>;
    pub type Device = burn::backend::ndarray::NdArrayDevice;

    pub fn get_device() -> Device {
        Device::Cpu
    }
}

#[cfg(not(any(feature = "metal", feature = "cuda", feature = "cpu")))]
compile_error!("Please select a backend by enabling one of: metal, cuda, or cpu features");

#[cfg(any(feature = "metal", feature = "cuda", feature = "cpu"))]
use backend::{get_device, Backend};

fn main() {
    #[cfg(not(any(feature = "metal", feature = "cuda", feature = "cpu")))]
    {
        eprintln!("Error: No backend feature enabled. Use --features metal, --features cuda, or --features cpu");
        std::process::exit(1);
    }

    #[cfg(any(feature = "metal", feature = "cuda", feature = "cpu"))]
    run_example();
}

#[cfg(any(feature = "metal", feature = "cuda", feature = "cpu"))]
fn run_example() {
    println!("LongCat Video Generation Example");
    println!("=================================\n");

    // Get device
    let device = get_device();
    println!("Using device: {:?}\n", device);

    // Check for model weights (prefer .bpk, fall back to .safetensors)
    let dit_weights = find_weights("models/longcat_dit");
    let vae_weights = find_weights("models/wan_vae");
    let text_weights = find_weights("models/umt5_xxl");

    // Build pipeline with weights if available
    let pipeline: LongCatPipeline<Backend> = if let (Some(dit), Some(vae), Some(text)) =
        (&dit_weights, &vae_weights, &text_weights)
    {
        println!("Loading pre-trained weights...");
        println!("  DiT: {}", dit.display());
        println!("  VAE: {}", vae.display());
        println!("  Text: {}", text.display());
        PipelineBuilder::new(device.clone())
            .with_dit_weights(dit)
            .with_vae_weights(vae)
            .with_text_weights(text)
            .build()
            .expect("Failed to build pipeline with weights")
    } else {
        println!("No pre-trained weights found. Using random initialization.");
        println!("For actual video generation, download weights to:");
        println!("  - models/longcat_dit.bpk (or .safetensors)");
        println!("  - models/wan_vae.bpk (or .safetensors)");
        println!("  - models/umt5_xxl.bpk (or .safetensors)\n");
        LongCatPipeline::new(&device)
    };

    // Print model info
    println!("Model Configuration:");
    println!("  - Layers: {}", pipeline.config().num_layers);
    println!("  - Hidden size: {}", pipeline.config().hidden_size);
    println!(
        "  - Approx. parameters: {:.1}B",
        pipeline.config().num_params() as f64 / 1e9
    );
    println!();

    // Example 1: Text-to-Video generation
    text_to_video_example(&pipeline);

    // Example 2: Fast generation (480p, fewer steps)
    fast_generation_example(&pipeline);

    // Example 3: High-quality generation (720p)
    // Note: Requires more VRAM
    // high_quality_example(&pipeline);

    println!("\nDone!");
}

/// Example: Generate video from text prompt
#[cfg(any(feature = "metal", feature = "cuda", feature = "cpu"))]
fn text_to_video_example(pipeline: &LongCatPipeline<Backend>) {
    println!("Example 1: Text-to-Video Generation");
    println!("------------------------------------");

    let prompt = "A cat playing with a ball of yarn in a sunny living room";
    println!("Prompt: \"{}\"", prompt);

    let config = GenerateConfig::default();
    println!(
        "Config: {}x{} @ {}fps, {} frames",
        config.width, config.height, config.fps, config.num_frames
    );
    println!(
        "Steps: {}, Guidance: {}",
        config.num_inference_steps, config.guidance_scale
    );

    println!("\nGenerating video...");
    let _video = pipeline.generate(prompt, &config);
    println!(
        "Generated video shape: [1, 3, {}, {}, {}]",
        config.num_frames, config.height, config.width
    );
    println!();
}

/// Example: Fast generation for quick previews
#[cfg(any(feature = "metal", feature = "cuda", feature = "cpu"))]
fn fast_generation_example(pipeline: &LongCatPipeline<Backend>) {
    println!("Example 2: Fast Generation (480p)");
    println!("----------------------------------");

    let prompt = "Ocean waves crashing on a beach at sunset";
    println!("Prompt: \"{}\"", prompt);

    let config = GenerateConfig::fast_480p();
    println!(
        "Config: {}x{} @ {}fps, {} frames",
        config.width, config.height, config.fps, config.num_frames
    );
    println!(
        "Steps: {}, Guidance: {}",
        config.num_inference_steps, config.guidance_scale
    );

    println!("\nGenerating video...");
    let _video = pipeline.generate(prompt, &config);
    println!(
        "Generated video shape: [1, 3, {}, {}, {}]",
        config.num_frames, config.height, config.width
    );
    println!();
}

/// Example: High-quality 720p generation
#[allow(dead_code)]
#[cfg(any(feature = "metal", feature = "cuda", feature = "cpu"))]
fn high_quality_example(pipeline: &LongCatPipeline<Backend>) {
    println!("Example 3: High-Quality Generation (720p)");
    println!("------------------------------------------");

    let prompt = "A beautiful mountain landscape with clouds rolling by";
    println!("Prompt: \"{}\"", prompt);

    let config = GenerateConfig::quality_720p();
    println!(
        "Config: {}x{} @ {}fps, {} frames",
        config.width, config.height, config.fps, config.num_frames
    );
    println!(
        "Steps: {}, Guidance: {}",
        config.num_inference_steps, config.guidance_scale
    );

    println!("\nGenerating video (this may take a while)...");
    let _video = pipeline.generate(prompt, &config);
    println!(
        "Generated video shape: [1, 3, {}, {}, {}]",
        config.num_frames, config.height, config.width
    );
    println!();
}
