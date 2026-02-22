//! LongCat DiT (Diffusion Transformer) Model
//!
//! 48-layer transformer for video generation with 3D spatial-temporal attention.
//! Matches the actual Kijai/ComfyUI safetensors structure.

use burn::module::{Ignored, Module};
use burn::nn::{Linear, LinearConfig, RmsNorm as BurnRmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::activation::silu;

use super::attention::{CrossAttention, SelfAttention3D};
use super::embeddings::{FinalLayer, PatchEmbed3D, TextEmbedding};
use super::feed_forward::FeedForward;
use super::modulation::TimestepEmbedder;
use super::rope::Rope3DFreqs;
use crate::LongCatConfig;

/// Single DiT Block
///
/// Contains:
/// 1. 3D Self-Attention with RoPE
/// 2. Cross-Attention for text conditioning
/// 3. SwiGLU Feed-Forward Network
/// 4. Single AdaLN modulation for all (6 params: 2 for each of self_attn, cross_attn, ffn)
#[derive(Module, Debug)]
pub struct DiTBlock<B: Backend> {
    /// 3D Self-attention
    self_attn: SelfAttention3D<B>,
    /// Cross-attention
    cross_attn: CrossAttention<B>,
    /// Pre-norm for FFN (only norm3 exists in the model)
    norm3: BurnRmsNorm<B>,
    /// Feed-forward network
    ffn: FeedForward<B>,
    /// Single AdaLN modulation for all (Sequential: SiLU, Linear)
    /// Output: 6 * hidden_size (shift/scale/gate for self_attn + cross_attn)
    modulation: Linear<B>,
    /// Hidden size for residual connections
    hidden_size: usize,
}

impl<B: Backend> DiTBlock<B> {
    /// Create a new DiT block
    pub fn new(config: &LongCatConfig, device: &B::Device) -> Self {
        // modulation.1 in model - output 24576 = 6 * 4096
        let modulation = LinearConfig::new(config.adaln_embedding_size, 6 * config.hidden_size)
            .with_bias(true)
            .init(device);

        Self {
            self_attn: SelfAttention3D::new(config, device),
            cross_attn: CrossAttention::new(config, device),
            norm3: RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.norm_eps)
                .init(device),
            ffn: FeedForward::new(config, device),
            modulation,
            hidden_size: config.hidden_size,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Hidden states [batch, seq_len, hidden_size]
    /// * `context` - Text embeddings [batch, text_len, text_hidden_size]
    /// * `t_emb` - Timestep embedding [batch, adaln_embed_size]
    /// * `rope_freqs` - Precomputed RoPE frequencies
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        context: Tensor<B, 3>,
        t_emb: Tensor<B, 2>,
        rope_freqs: Option<&Rope3DFreqs<B>>,
    ) -> Tensor<B, 3> {
        let [batch, _seq, _hidden] = x.dims();
        let hidden_size = self.hidden_size;

        // Apply modulation: SiLU then Linear
        let mod_out = self.modulation.forward(silu(t_emb));

        // Split into 6 parts: (shift_s, scale_s, gate_s, shift_c, scale_c, gate_c)
        let shift_s = mod_out.clone().slice([0..batch, 0..hidden_size]);
        let scale_s = mod_out.clone().slice([0..batch, hidden_size..2 * hidden_size]);
        let gate_s = mod_out.clone().slice([0..batch, 2 * hidden_size..3 * hidden_size]);
        let shift_c = mod_out.clone().slice([0..batch, 3 * hidden_size..4 * hidden_size]);
        let scale_c = mod_out.clone().slice([0..batch, 4 * hidden_size..5 * hidden_size]);
        let gate_c = mod_out.slice([0..batch, 5 * hidden_size..6 * hidden_size]);

        // Self-attention with modulation (no pre-norm, modulation applied inside attention or before)
        let x_mod = x.clone() * (scale_s.clone().unsqueeze_dim(1) + 1.0) + shift_s.unsqueeze_dim(1);
        let self_attn_out = self.self_attn.forward(x_mod, rope_freqs);
        let x = x + gate_s.unsqueeze_dim(1) * self_attn_out;

        // Cross-attention with modulation
        let x_mod = x.clone() * (scale_c.clone().unsqueeze_dim(1) + 1.0) + shift_c.unsqueeze_dim(1);
        let cross_attn_out = self.cross_attn.forward(x_mod, context);
        let x = x + gate_c.unsqueeze_dim(1) * cross_attn_out;

        // FFN with norm3 (the only explicit norm in model)
        let x_norm = self.norm3.forward(x.clone());
        let ffn_out = self.ffn.forward(x_norm);
        let x = x + ffn_out;

        x
    }
}

/// LongCat DiT Model
///
/// Full 48-layer diffusion transformer for video generation.
/// Matches the actual Kijai/ComfyUI safetensors structure.
#[derive(Module, Debug)]
pub struct LongCatDiT<B: Backend> {
    /// Patch embedding (3D conv): patch_embedding
    patch_embedding: PatchEmbed3D<B>,
    /// Time embedding MLP: time_embedding.mlp.0/2
    time_embedding: TimestepEmbedder<B>,
    /// Text embedding MLP: text_embedding.0/2
    text_embedding: TextEmbedding<B>,
    /// Transformer blocks: blocks.0-47
    blocks: Vec<DiTBlock<B>>,
    /// Final output layer: head.head, head.modulation
    head: FinalLayer<B>,
    /// Model configuration (not a module parameter)
    config: Ignored<LongCatConfig>,
}

impl<B: Backend> LongCatDiT<B> {
    /// Initialize the model from config
    pub fn new(config: &LongCatConfig, device: &B::Device) -> Self {
        let blocks = (0..config.num_layers)
            .map(|_| DiTBlock::new(config, device))
            .collect();

        Self {
            patch_embedding: PatchEmbed3D::new(config, device),
            time_embedding: TimestepEmbedder::new(config, device),
            text_embedding: TextEmbedding::new(config, device),
            blocks,
            head: FinalLayer::new(config, device),
            config: Ignored(config.clone()),
        }
    }

    /// Forward pass - predict velocity field
    ///
    /// # Arguments
    /// * `latents` - Noisy latents [batch, channels, time, height, width]
    /// * `timestep` - Diffusion timestep [batch]
    /// * `text_embeds` - Text encoder outputs [batch, text_len, text_hidden_size]
    ///
    /// # Returns
    /// Predicted velocity [batch, channels, time, height, width]
    pub fn forward(
        &self,
        latents: Tensor<B, 5>,
        timestep: Tensor<B, 1>,
        text_embeds: Tensor<B, 3>,
    ) -> Tensor<B, 5> {
        let device = latents.device();
        let [_batch, _channels, time, height, width] = latents.dims();

        // Get patch dimensions
        let (t_patches, h_patches, w_patches) = self.patch_embedding.output_shape(time, height, width);

        // Embed timestep
        let t_emb = self.time_embedding.forward(timestep);

        // Embed text (project from text_hidden_size to hidden_size)
        let text_embeds = self.text_embedding.forward(text_embeds);

        // Convert latents to patch tokens
        let x = self.patch_embedding.forward(latents);

        // Precompute RoPE frequencies
        let rope_freqs = Rope3DFreqs::precompute(
            t_patches,
            h_patches,
            w_patches,
            self.config.0.head_dim,
            self.config.0.rope_theta,
            &device,
        );

        // Pass through transformer blocks
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x, text_embeds.clone(), t_emb.clone(), Some(&rope_freqs));
        }

        // Final layer - unpatchify
        self.head.forward(x, t_emb, t_patches, h_patches, w_patches)
    }

    /// Get model configuration
    pub fn config(&self) -> &LongCatConfig {
        &self.config.0
    }
}

impl LongCatConfig {
    /// Initialize the DiT model
    pub fn init<B: Backend>(&self, device: &B::Device) -> LongCatDiT<B> {
        LongCatDiT::new(self, device)
    }
}
