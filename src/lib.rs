//! LongCat Video Generation Model in Burn
//!
//! LongCat is a 13.6B parameter Diffusion Transformer (DiT) for video generation,
//! supporting text-to-video (T2V), image-to-video (I2V), and video continuation.
//!
//! ## Architecture
//!
//! - **DiT**: 48 transformer layers with 3D spatial-temporal attention
//! - **Text Encoder**: UMT5-XXL for multilingual text understanding
//! - **VAE**: WAN 2.1 VAE for video encoding/decoding (4x8x8 compression, 4x16x16 with patchify)
//! - **Conditioning**: Cross-attention with KV caching for video continuation
//!
//! ## Key Features
//!
//! - Flow matching diffusion (velocity prediction)
//! - 3D RoPE positional encoding
//! - Block sparse attention for efficiency
//! - AdaLN-Zero modulation
//! - RMSNorm + QKNorm for stability
//!
//! ## Usage
//!
//! ```rust,ignore
//! use longcat_burn::{LongCatPipeline, GenerateConfig};
//!
//! // Create pipeline with pre-trained weights
//! let pipeline = LongCatPipeline::builder(device)
//!     .with_dit_weights("longcat_dit.safetensors")
//!     .with_vae_weights("wan_vae.safetensors")
//!     .with_text_weights("umt5_xxl.safetensors")
//!     .build()?;
//!
//! // Generate video
//! let config = GenerateConfig::default();
//! let video = pipeline.generate("A cat playing in a garden", &config);
//! ```

pub mod config;
pub mod modules;
pub mod load;
pub mod scheduler;
pub mod pipeline;

// Re-export main types
pub use config::{LongCatConfig, GenerateConfig};
pub use modules::dit::LongCatDiT;
pub use modules::vae::{WanVae, WanVaeConfig};
pub use modules::memory::{get_attention_slice_size, set_attention_slice_size, MemoryConfig};
pub use scheduler::{FlowMatchingScheduler, SchedulerConfig};
pub use pipeline::{LongCatPipeline, PipelineBuilder, PipelineBuildError, MemoryEstimate, GenerationControl, GenerationProgress};
