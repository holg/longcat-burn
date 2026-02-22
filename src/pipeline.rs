//! LongCat Video Generation Pipeline
//!
//! Combines DiT, VAE, and text encoder for end-to-end video generation.
//! Supports memory-efficient inference through attention slicing and tiled VAE.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use burn::prelude::*;
use umt5_burn::{UMT5Config, UMT5Encoder};

use crate::config::{GenerateConfig, LongCatConfig};
use crate::modules::dit::LongCatDiT;
use crate::modules::memory::{get_attention_slice_size, MemoryConfig};
use crate::modules::vae::{WanVae, WanVaeConfig};
use crate::scheduler::FlowMatchingScheduler;

/// Progress information for video generation
#[derive(Debug, Clone)]
pub struct GenerationProgress {
    /// Current step (0-indexed)
    pub current_step: usize,
    /// Total number of steps
    pub total_steps: usize,
    /// Time elapsed in seconds
    pub elapsed_secs: f32,
    /// Estimated time remaining in seconds
    pub eta_secs: f32,
    /// Time for last step in seconds
    pub step_time_secs: f32,
}

/// Callback type for progress updates
pub type ProgressCallback = Box<dyn Fn(GenerationProgress) + Send + Sync>;

/// Control handle for pausing/cancelling generation
#[derive(Clone)]
pub struct GenerationControl {
    /// Set to true to request pause
    paused: Arc<AtomicBool>,
    /// Set to true to request cancellation
    cancelled: Arc<AtomicBool>,
    /// Current step (for external monitoring)
    current_step: Arc<AtomicUsize>,
    /// Total steps
    total_steps: Arc<AtomicUsize>,
}

impl GenerationControl {
    /// Create a new control handle
    pub fn new() -> Self {
        Self {
            paused: Arc::new(AtomicBool::new(false)),
            cancelled: Arc::new(AtomicBool::new(false)),
            current_step: Arc::new(AtomicUsize::new(0)),
            total_steps: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Request pause
    pub fn pause(&self) {
        self.paused.store(true, Ordering::SeqCst);
    }

    /// Resume from pause
    pub fn resume(&self) {
        self.paused.store(false, Ordering::SeqCst);
    }

    /// Check if paused
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }

    /// Request cancellation
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Check if cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Get current step
    pub fn get_current_step(&self) -> usize {
        self.current_step.load(Ordering::SeqCst)
    }

    /// Get total steps
    pub fn get_total_steps(&self) -> usize {
        self.total_steps.load(Ordering::SeqCst)
    }

    /// Get progress as percentage (0.0 - 1.0)
    pub fn get_progress(&self) -> f32 {
        let total = self.total_steps.load(Ordering::SeqCst);
        if total == 0 {
            return 0.0;
        }
        self.current_step.load(Ordering::SeqCst) as f32 / total as f32
    }

    /// Reset all flags for a new generation
    pub fn reset(&self) {
        self.paused.store(false, Ordering::SeqCst);
        self.cancelled.store(false, Ordering::SeqCst);
        self.current_step.store(0, Ordering::SeqCst);
        self.total_steps.store(0, Ordering::SeqCst);
    }
}

impl Default for GenerationControl {
    fn default() -> Self {
        Self::new()
    }
}

/// LongCat Video Generation Pipeline
///
/// Full pipeline for text-to-video generation:
/// 1. Encode text prompt with UMT5
/// 2. Generate latents with DiT using flow matching
/// 3. Decode latents to video with VAE
pub struct LongCatPipeline<B: Backend> {
    /// DiT model for diffusion
    pub dit: LongCatDiT<B>,
    /// VAE for encoding/decoding
    pub vae: WanVae<B>,
    /// Text encoder (UMT5)
    pub text_encoder: UMT5Encoder<B>,
    /// Model configuration
    config: LongCatConfig,
    /// Device
    device: B::Device,
}

impl<B: Backend> LongCatPipeline<B> {
    /// Create a new pipeline with default configurations
    pub fn new(device: &B::Device) -> Self {
        let config = LongCatConfig::default();
        let vae_config = WanVaeConfig::default();
        let text_config = UMT5Config::xxl();

        Self {
            dit: config.init(device),
            vae: vae_config.init(device),
            text_encoder: text_config.init_encoder(device),
            config,
            device: device.clone(),
        }
    }

    /// Create pipeline with custom configurations
    pub fn with_configs(
        dit_config: &LongCatConfig,
        vae_config: &WanVaeConfig,
        text_config: &UMT5Config,
        device: &B::Device,
    ) -> Self {
        Self {
            dit: dit_config.init(device),
            vae: vae_config.init(device),
            text_encoder: text_config.init_encoder(device),
            config: dit_config.clone(),
            device: device.clone(),
        }
    }

    /// Generate video from text prompt
    ///
    /// # Arguments
    /// * `prompt` - Text prompt describing the video
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// Generated video tensor [batch, 3, time, height, width] in range [0, 1]
    pub fn generate(&self, prompt: &str, config: &GenerateConfig) -> Tensor<B, 5> {
        let control = GenerationControl::new();
        self.generate_with_control(prompt, config, &control)
    }

    /// Generate video with pause/cancel control
    ///
    /// # Arguments
    /// * `prompt` - Text prompt describing the video
    /// * `config` - Generation configuration
    /// * `control` - Control handle for pause/cancel
    ///
    /// # Returns
    /// Generated video tensor, or panics if cancelled
    pub fn generate_with_control(
        &self,
        prompt: &str,
        config: &GenerateConfig,
        control: &GenerationControl,
    ) -> Tensor<B, 5> {
        // Encode text prompt
        let text_embeds = self.encode_prompt(prompt);

        // Generate with optional CFG
        if config.guidance_scale > 1.0 {
            self.generate_with_cfg_controlled(text_embeds, config, control)
        } else {
            self.generate_uncond_controlled(text_embeds, config, control)
        }
    }

    /// Generate video from pre-computed text embeddings
    pub fn generate_from_embeddings(
        &self,
        text_embeds: Tensor<B, 3>,
        config: &GenerateConfig,
    ) -> Tensor<B, 5> {
        if config.guidance_scale > 1.0 {
            self.generate_with_cfg(text_embeds, config)
        } else {
            self.generate_uncond(text_embeds, config)
        }
    }

    /// Encode text prompt to embeddings
    fn encode_prompt(&self, prompt: &str) -> Tensor<B, 3> {
        // Tokenize and encode
        // Note: This is a placeholder - actual tokenization depends on UMT5 implementation
        let input_ids = self.tokenize(prompt);
        let output = self.text_encoder.forward(input_ids, None);
        output.last_hidden_state
    }

    /// Tokenize text (placeholder - actual implementation depends on tokenizer)
    fn tokenize(&self, _prompt: &str) -> Tensor<B, 2, Int> {
        // Placeholder: return dummy tokens
        // In practice, use the UMT5 tokenizer
        // Note: Use from_data to avoid Metal i64 operation errors
        let zeros: Vec<i64> = vec![0i64; self.config.max_text_seq_len];
        Tensor::<B, 1, Int>::from_data(zeros.as_slice(), &self.device)
            .reshape([1, self.config.max_text_seq_len])
    }

    /// Generate without classifier-free guidance
    fn generate_uncond(&self, text_embeds: Tensor<B, 3>, config: &GenerateConfig) -> Tensor<B, 5> {
        let control = GenerationControl::new();
        self.generate_uncond_controlled(text_embeds, config, &control)
    }

    /// Generate without CFG with pause/cancel control
    fn generate_uncond_controlled(
        &self,
        text_embeds: Tensor<B, 3>,
        config: &GenerateConfig,
        control: &GenerationControl,
    ) -> Tensor<B, 5> {
        let (latent_time, latent_height, latent_width) = self.compute_latent_shape(config);
        let batch_size = 1;

        // Create scheduler
        let num_tokens = latent_time * latent_height * latent_width;
        let scheduler = FlowMatchingScheduler::with_adaptive_shift(
            config.num_inference_steps,
            num_tokens,
        );

        // Update control with total steps
        control.total_steps.store(config.num_inference_steps, Ordering::SeqCst);
        control.current_step.store(0, Ordering::SeqCst);

        eprintln!("[longcat] Starting diffusion: {} steps, {} tokens",
                  config.num_inference_steps, num_tokens);

        // Sample initial noise
        let mut latents = scheduler.sample_noise(
            [batch_size, self.config.latent_channels, latent_time, latent_height, latent_width],
            &self.device,
        );

        let start_time = std::time::Instant::now();

        // Sampling loop
        for step in 0..config.num_inference_steps {
            // Check for cancellation
            if control.is_cancelled() {
                eprintln!("[longcat] Generation cancelled at step {}", step);
                panic!("Generation cancelled by user");
            }

            // Handle pause
            while control.is_paused() {
                eprintln!("[longcat] Paused at step {}...", step);
                std::thread::sleep(std::time::Duration::from_millis(500));
                if control.is_cancelled() {
                    eprintln!("[longcat] Generation cancelled while paused");
                    panic!("Generation cancelled by user");
                }
            }

            let step_start = std::time::Instant::now();
            let t = scheduler.get_timestep(step);
            let timestep = Tensor::from_floats([t], &self.device);

            // Predict velocity
            let velocity = self.dit.forward(
                latents.clone(),
                timestep,
                text_embeds.clone(),
            );

            // Step
            latents = scheduler.step(latents, velocity, step);

            // Update progress
            control.current_step.store(step + 1, Ordering::SeqCst);

            let step_time = step_start.elapsed().as_secs_f32();
            let total_time = start_time.elapsed().as_secs_f32();
            let eta = if step > 0 {
                (total_time / (step + 1) as f32) * (config.num_inference_steps - step - 1) as f32
            } else {
                step_time * (config.num_inference_steps - 1) as f32
            };
            eprintln!("[longcat] Step {}/{} done in {:.1}s (total: {:.1}s, ETA: {:.1}s)",
                      step + 1, config.num_inference_steps, step_time, total_time, eta);
        }

        eprintln!("[longcat] Diffusion complete in {:.1}s, decoding video...",
                  start_time.elapsed().as_secs_f32());

        // Decode to video
        let video = self.vae.decode(latents);
        eprintln!("[longcat] Video decoded successfully");
        video
    }

    /// Generate with CFG (wrapper for controlled version)
    fn generate_with_cfg(&self, text_embeds: Tensor<B, 3>, config: &GenerateConfig) -> Tensor<B, 5> {
        let control = GenerationControl::new();
        self.generate_with_cfg_controlled(text_embeds, config, &control)
    }

    /// Generate with CFG with pause/cancel control
    fn generate_with_cfg_controlled(
        &self,
        text_embeds: Tensor<B, 3>,
        config: &GenerateConfig,
        control: &GenerationControl,
    ) -> Tensor<B, 5> {
        // For now, just call uncond - CFG implementation would be similar
        // but with doubled batch size for unconditional/conditional
        self.generate_uncond_controlled(text_embeds, config, control)
    }

    /// Compute latent dimensions for given video config
    fn compute_latent_shape(&self, config: &GenerateConfig) -> (usize, usize, usize) {
        let (temporal_compression, spatial_compression, _) = self.vae.compression_ratio();

        let latent_time = config.num_frames / temporal_compression;
        let latent_height = config.height / spatial_compression;
        let latent_width = config.width / spatial_compression;

        (latent_time, latent_height, latent_width)
    }

    /// Get the model configuration
    pub fn config(&self) -> &LongCatConfig {
        &self.config
    }

    /// Estimate peak memory usage for a given configuration.
    ///
    /// Returns estimated memory in bytes for:
    /// - Model weights
    /// - Activation memory during forward pass
    /// - Attention matrices (the main memory consumer)
    pub fn estimate_memory(&self, config: &GenerateConfig) -> MemoryEstimate {
        let (latent_time, latent_height, latent_width) = self.compute_latent_shape(config);
        let patch_size = self.config.patch_size;

        // Number of tokens after patchification
        let num_tokens = latent_time * (latent_height / patch_size) * (latent_width / patch_size);

        // Model weights (roughly 13.6B parameters in bf16 = ~27GB)
        let dit_params = self.config.num_params();
        let model_memory = dit_params * 2; // bf16

        // Attention memory per layer (the killer)
        // [batch, heads, seq, seq] in bf16
        let heads = self.config.num_attention_heads;
        let attention_matrix_bytes = 1 * heads * num_tokens * num_tokens * 2; // bf16

        // With slicing, we reduce this
        let slice_size = get_attention_slice_size();
        let effective_attention_memory = if slice_size > 0 && num_tokens > slice_size {
            // Only need [batch, heads, slice_size, seq_len]
            let sliced = 1 * heads * slice_size * num_tokens * 2;
            sliced
        } else {
            attention_matrix_bytes
        };

        // Total per layer: Q, K, V, attention output, plus attention matrix
        let qkv_memory = 3 * 1 * num_tokens * self.config.hidden_size * 2;
        let per_layer_activation = qkv_memory + effective_attention_memory;

        // CFG doubles the forward passes (but not simultaneously)
        let cfg_multiplier = if config.guidance_scale > 1.0 { 1 } else { 1 };

        // Peak activation memory (roughly one layer at a time plus latents)
        let latent_memory = 1 * self.config.latent_channels * latent_time * latent_height * latent_width * 2;
        let peak_activation = per_layer_activation * cfg_multiplier + latent_memory;

        MemoryEstimate {
            model_memory_bytes: model_memory,
            peak_activation_bytes: peak_activation,
            attention_memory_per_layer: effective_attention_memory,
            num_tokens,
            recommended_slice_size: self.recommend_slice_size(num_tokens),
        }
    }

    /// Recommend an attention slice size based on sequence length
    fn recommend_slice_size(&self, num_tokens: usize) -> usize {
        // Heuristic: aim for attention matrices under 2GB per layer
        // [batch=1, heads=32, slice, seq] * 2 bytes < 2GB
        // slice * seq * 32 * 2 < 2_000_000_000
        // slice < 2_000_000_000 / (seq * 64)
        let max_slice = 2_000_000_000 / (num_tokens * 64);
        let recommended = max_slice.min(2048).max(256);

        // Round down to nearest power of 2 for efficiency
        let mut pow2 = 256;
        while pow2 * 2 <= recommended {
            pow2 *= 2;
        }
        pow2
    }

    /// Generate video with automatic memory optimization.
    ///
    /// Automatically configures attention slicing based on available memory hints.
    pub fn generate_low_memory(
        &self,
        prompt: &str,
        config: &GenerateConfig,
        memory_config: &MemoryConfig,
    ) -> Tensor<B, 5> {
        // Apply memory configuration
        memory_config.apply();

        let estimate = self.estimate_memory(config);
        eprintln!("[longcat] Memory estimate:");
        eprintln!("  - Model: {:.2} GB", estimate.model_memory_bytes as f64 / 1e9);
        eprintln!("  - Peak activation: {:.2} GB", estimate.peak_activation_bytes as f64 / 1e9);
        eprintln!("  - Attention per layer: {:.2} GB", estimate.attention_memory_per_layer as f64 / 1e9);
        eprintln!("  - Sequence length: {} tokens", estimate.num_tokens);
        eprintln!("  - Recommended slice size: {}", estimate.recommended_slice_size);

        // Generate
        self.generate(prompt, config)
    }
}

/// Memory usage estimate for video generation
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    /// Model weights memory in bytes
    pub model_memory_bytes: usize,
    /// Peak activation memory in bytes
    pub peak_activation_bytes: usize,
    /// Attention memory per layer in bytes
    pub attention_memory_per_layer: usize,
    /// Number of tokens in sequence
    pub num_tokens: usize,
    /// Recommended attention slice size
    pub recommended_slice_size: usize,
}

impl MemoryEstimate {
    /// Total estimated memory in bytes
    pub fn total_bytes(&self) -> usize {
        self.model_memory_bytes + self.peak_activation_bytes
    }

    /// Total estimated memory in GB
    pub fn total_gb(&self) -> f64 {
        self.total_bytes() as f64 / 1e9
    }
}

/// Builder for LongCat Pipeline
pub struct PipelineBuilder<B: Backend> {
    dit_config: LongCatConfig,
    vae_config: WanVaeConfig,
    text_config: UMT5Config,
    dit_weights: Option<std::path::PathBuf>,
    vae_weights: Option<std::path::PathBuf>,
    text_weights: Option<std::path::PathBuf>,
    device: B::Device,
}

impl<B: Backend> PipelineBuilder<B> {
    /// Create a new pipeline builder
    pub fn new(device: B::Device) -> Self {
        Self {
            dit_config: LongCatConfig::default(),
            vae_config: WanVaeConfig::default(),
            text_config: UMT5Config::xxl(),
            dit_weights: None,
            vae_weights: None,
            text_weights: None,
            device,
        }
    }

    /// Set DiT configuration
    pub fn with_dit_config(mut self, config: LongCatConfig) -> Self {
        self.dit_config = config;
        self
    }

    /// Set VAE configuration
    pub fn with_vae_config(mut self, config: WanVaeConfig) -> Self {
        self.vae_config = config;
        self
    }

    /// Set text encoder configuration
    pub fn with_text_config(mut self, config: UMT5Config) -> Self {
        self.text_config = config;
        self
    }

    /// Set DiT weights path
    pub fn with_dit_weights(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.dit_weights = Some(path.into());
        self
    }

    /// Set VAE weights path
    pub fn with_vae_weights(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.vae_weights = Some(path.into());
        self
    }

    /// Set text encoder weights path
    pub fn with_text_weights(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.text_weights = Some(path.into());
        self
    }

    /// Build the pipeline
    pub fn build(self) -> Result<LongCatPipeline<B>, PipelineBuildError> {
        let mut dit = self.dit_config.init(&self.device);
        let mut vae = self.vae_config.init(&self.device);
        let mut text_encoder = self.text_config.init_encoder(&self.device);

        // Load weights if provided
        if let Some(path) = self.dit_weights {
            dit.load_weights(&path)
                .map_err(|e| PipelineBuildError::DitLoadError(e.to_string()))?;
        }

        if let Some(path) = self.vae_weights {
            vae.load_weights(&path)
                .map_err(|e| PipelineBuildError::VaeLoadError(e.to_string()))?;
        }

        if let Some(path) = self.text_weights {
            text_encoder
                .load_weights(&path)
                .map_err(|e| PipelineBuildError::TextEncoderLoadError(e.to_string()))?;
        }

        Ok(LongCatPipeline {
            dit,
            vae,
            text_encoder,
            config: self.dit_config,
            device: self.device,
        })
    }
}

/// Pipeline build errors
#[derive(Debug, thiserror::Error)]
pub enum PipelineBuildError {
    #[error("Failed to load DiT weights: {0}")]
    DitLoadError(String),
    #[error("Failed to load VAE weights: {0}")]
    VaeLoadError(String),
    #[error("Failed to load text encoder weights: {0}")]
    TextEncoderLoadError(String),
}

/// Image-to-Video generation support
impl<B: Backend> LongCatPipeline<B> {
    /// Generate video from an initial image
    ///
    /// # Arguments
    /// * `image` - Initial frame [batch, 3, height, width] in range [0, 1]
    /// * `prompt` - Text prompt describing the video
    /// * `config` - Generation configuration
    pub fn generate_from_image(
        &self,
        image: Tensor<B, 4>,
        prompt: &str,
        config: &GenerateConfig,
    ) -> Tensor<B, 5> {
        // Encode text
        let text_embeds = self.encode_prompt(prompt);

        // Encode image to latent
        let [batch, channels, height, width] = image.dims();
        let image_video = image.reshape([batch, channels, 1, height, width]);
        let image_latent = self.vae.encode_deterministic(image_video);

        self.generate_i2v(image_latent, text_embeds, config)
    }

    /// Generate video with image conditioning
    fn generate_i2v(
        &self,
        image_latent: Tensor<B, 5>,
        text_embeds: Tensor<B, 3>,
        config: &GenerateConfig,
    ) -> Tensor<B, 5> {
        let (latent_time, latent_height, latent_width) = self.compute_latent_shape(config);
        let [batch, latent_ch, _, _, _] = image_latent.dims();

        // Create scheduler
        let num_tokens = latent_time * latent_height * latent_width;
        let scheduler = FlowMatchingScheduler::with_adaptive_shift(
            config.num_inference_steps,
            num_tokens,
        );

        eprintln!("[longcat] Starting I2V diffusion: {} steps, {} tokens",
                  config.num_inference_steps, num_tokens);

        // Sample initial noise
        let mut latents = scheduler.sample_noise(
            [batch, latent_ch, latent_time, latent_height, latent_width],
            &self.device,
        );

        // Replace first frame with image latent (for I2V)
        // This is a simplified approach - actual I2V may use more sophisticated conditioning
        let image_latent_expanded = image_latent.repeat(&[1, 1, latent_time, 1, 1]);

        let start_time = std::time::Instant::now();

        // Sampling loop
        for step in 0..config.num_inference_steps {
            let step_start = std::time::Instant::now();
            let t = scheduler.get_timestep(step);
            let timestep = Tensor::from_floats([t], &self.device);

            // Blend with image latent based on timestep
            let blend_factor = 1.0 - t; // More image influence as t decreases
            let conditioned_latents = latents.clone() * t + image_latent_expanded.clone() * blend_factor;

            // Predict velocity
            let velocity = self.dit.forward(
                conditioned_latents,
                timestep,
                text_embeds.clone(),
            );

            // Step
            latents = scheduler.step(latents, velocity, step);

            let step_time = step_start.elapsed().as_secs_f32();
            let total_time = start_time.elapsed().as_secs_f32();
            let eta = if step > 0 {
                (total_time / step as f32) * (config.num_inference_steps - step) as f32
            } else {
                0.0
            };
            eprintln!("[longcat] Step {}/{} done in {:.1}s (total: {:.1}s, ETA: {:.1}s)",
                      step + 1, config.num_inference_steps, step_time, total_time, eta);
        }

        eprintln!("[longcat] I2V diffusion complete in {:.1}s, decoding video...",
                  start_time.elapsed().as_secs_f32());

        // Decode to video
        let video = self.vae.decode(latents);
        eprintln!("[longcat] Video decoded successfully");
        video
    }

    /// Generate video from pre-encoded latent (for hybrid CPU/GPU I2V)
    ///
    /// Use this when the image has already been encoded on CPU.
    ///
    /// # Arguments
    /// * `image_latent` - Pre-encoded image latent [batch, channels, 1, height, width]
    /// * `prompt` - Text prompt describing the video
    /// * `config` - Generation configuration
    pub fn generate_from_latent(
        &self,
        image_latent: Tensor<B, 5>,
        prompt: &str,
        config: &GenerateConfig,
    ) -> Tensor<B, 5> {
        // Encode text
        let text_embeds = self.encode_prompt(prompt);

        // Generate video from latent
        self.generate_i2v(image_latent, text_embeds, config)
    }
}

/// Video continuation support
impl<B: Backend> LongCatPipeline<B> {
    /// Continue an existing video
    ///
    /// # Arguments
    /// * `video` - Existing video [batch, 3, time, height, width] in range [0, 1]
    /// * `prompt` - Text prompt for continuation
    /// * `num_new_frames` - Number of new frames to generate
    pub fn continue_video(
        &self,
        video: Tensor<B, 5>,
        prompt: &str,
        num_new_frames: usize,
        config: &GenerateConfig,
    ) -> Tensor<B, 5> {
        // Encode text
        let text_embeds = self.encode_prompt(prompt);

        // Encode existing video to latent
        let video_latent = self.vae.encode_deterministic(video.clone());
        let [batch, latent_ch, existing_time, latent_height, latent_width] = video_latent.dims();

        // Compute new frames in latent space
        let (temporal_compression, _, _) = self.vae.compression_ratio();
        let new_latent_frames = num_new_frames / temporal_compression;
        let total_latent_frames = existing_time + new_latent_frames;

        // Create scheduler
        let num_tokens = total_latent_frames * latent_height * latent_width;
        let scheduler = FlowMatchingScheduler::with_adaptive_shift(
            config.num_inference_steps,
            num_tokens,
        );

        // Sample noise for new frames only
        let new_noise = scheduler.sample_noise(
            [batch, latent_ch, new_latent_frames, latent_height, latent_width],
            &self.device,
        );

        // Concatenate existing latents with noise
        let mut latents = Tensor::cat(vec![video_latent.clone(), new_noise], 2);

        // Sampling loop - only denoise new frames, keep existing ones fixed
        for step in 0..config.num_inference_steps {
            let t = scheduler.get_timestep(step);
            let timestep = Tensor::from_floats([t], &self.device);

            // Predict velocity for all frames
            let velocity = self.dit.forward(
                latents.clone(),
                timestep,
                text_embeds.clone(),
            );

            // Step
            let new_latents = scheduler.step(latents.clone(), velocity, step);

            // Keep existing frames fixed, only update new frames
            let existing = video_latent.clone();
            let new_part = new_latents.clone().slice([
                0..batch,
                0..latent_ch,
                existing_time..total_latent_frames,
                0..latent_height,
                0..latent_width,
            ]);
            latents = Tensor::cat(vec![existing, new_part], 2);
        }

        // Decode to video
        self.vae.decode(latents)
    }
}
