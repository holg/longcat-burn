//! Patch embeddings for LongCat
//!
//! Converts video latents into patch tokens for transformer processing.
//! Matches the actual Kijai/ComfyUI safetensors structure.

use burn::module::{Ignored, Module};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::silu;

use crate::LongCatConfig;

/// 3D Patch Embedding Layer (matches actual model)
///
/// Uses 3D convolution: patch_embedding.weight [4096, 16, 1, 2, 2]
/// Converts 3D video latents [B, C, T, H, W] into patch tokens [B, num_patches, hidden_size].
#[derive(Module, Debug)]
pub struct PatchEmbed3D<B: Backend> {
    /// 3D convolution: [hidden_size, latent_channels, temporal_patch, spatial_patch, spatial_patch]
    weight: burn::module::Param<Tensor<B, 5>>,
    /// Bias: [hidden_size]
    bias: burn::module::Param<Tensor<B, 1>>,
    /// Patch size for spatial dimensions
    patch_size: Ignored<usize>,
    /// Patch size for temporal dimension
    temporal_patch_size: Ignored<usize>,
    /// Number of input channels
    in_channels: Ignored<usize>,
    /// Hidden size
    hidden_size: Ignored<usize>,
}

impl<B: Backend> PatchEmbed3D<B> {
    /// Create a new 3D patch embedding layer
    pub fn new(config: &LongCatConfig, device: &B::Device) -> Self {
        // patch_embedding.weight: [4096, 16, 1, 2, 2]
        let weight = Tensor::zeros(
            [config.hidden_size, config.latent_channels, config.temporal_patch_size, config.patch_size, config.patch_size],
            device,
        );
        let bias = Tensor::zeros([config.hidden_size], device);

        Self {
            weight: burn::module::Param::from_tensor(weight),
            bias: burn::module::Param::from_tensor(bias),
            patch_size: Ignored(config.patch_size),
            temporal_patch_size: Ignored(config.temporal_patch_size),
            in_channels: Ignored(config.latent_channels),
            hidden_size: Ignored(config.hidden_size),
        }
    }

    /// Convert latents to patch embeddings using 3D conv
    ///
    /// # Arguments
    /// * `x` - Input latents [batch, channels, time, height, width]
    ///
    /// # Returns
    /// Patch embeddings [batch, num_patches, hidden_size]
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 3> {
        let [batch, _channels, time, height, width] = x.dims();

        // Apply 3D convolution with stride = kernel_size (non-overlapping patches)
        // This is equivalent to unfold + linear projection
        let t_patches = time / self.temporal_patch_size.0;
        let h_patches = height / self.patch_size.0;
        let w_patches = width / self.patch_size.0;

        // Manual 3D conv: extract patches and project
        // For simplicity, reshape patches and use matrix multiply
        let x = x.reshape([
            batch,
            self.in_channels.0,
            t_patches,
            self.temporal_patch_size.0,
            h_patches,
            self.patch_size.0,
            w_patches,
            self.patch_size.0,
        ]);

        // Permute to [B, t_p, h_p, w_p, C, tp_size, p_size, p_size]
        let x = x.permute([0, 2, 4, 6, 1, 3, 5, 7]);

        // Flatten patches: [B, num_patches, patch_dim]
        let num_patches = t_patches * h_patches * w_patches;
        let patch_dim = self.in_channels.0 * self.temporal_patch_size.0 * self.patch_size.0 * self.patch_size.0;
        let x = x.reshape([batch, num_patches, patch_dim]);

        // Weight is [hidden, in_c, t, h, w] -> reshape to [patch_dim, hidden]
        // For batch matmul: [B, N, patch_dim] @ [patch_dim, hidden] = [B, N, hidden]
        let weight = self.weight.val().reshape([self.hidden_size.0, patch_dim]).transpose();

        // Linear projection using batched matmul
        // x: [B, N, patch_dim], weight: [patch_dim, hidden]
        // We need to broadcast weight for batch matmul
        let out = x.matmul(weight.unsqueeze_dim(0).repeat(&[batch, 1, 1]));
        out + self.bias.val().unsqueeze_dims(&[0, 1])
    }

    /// Get output dimensions for given input shape
    pub fn output_shape(&self, time: usize, height: usize, width: usize) -> (usize, usize, usize) {
        (
            time / self.temporal_patch_size.0,
            height / self.patch_size.0,
            width / self.patch_size.0,
        )
    }
}

/// Text Embedding MLP (matches actual model)
///
/// Projects text embeddings: text_embedding.0/2
/// Sequential(Linear, SiLU, Linear)
#[derive(Module, Debug)]
pub struct TextEmbedding<B: Backend> {
    /// First linear layer: text_embedding.0
    linear1: Linear<B>,
    /// Second linear layer: text_embedding.2
    linear2: Linear<B>,
}

impl<B: Backend> TextEmbedding<B> {
    /// Create text embedding MLP
    pub fn new(config: &LongCatConfig, device: &B::Device) -> Self {
        Self {
            linear1: LinearConfig::new(config.text_hidden_size, config.hidden_size)
                .with_bias(true)
                .init(device),
            linear2: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(true)
                .init(device),
        }
    }

    /// Forward pass
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(x);
        let x = silu(x);
        self.linear2.forward(x)
    }
}

/// Final Layer - unpatchify and project back to latent space
/// Matches actual model: head.head, head.modulation.1
#[derive(Module, Debug)]
pub struct FinalLayer<B: Backend> {
    /// Output projection: head.head [64, 4096]
    head: Linear<B>,
    /// AdaLN modulation: head.modulation.1 [8192, 512] (after SiLU)
    modulation: Linear<B>,
    /// Patch size
    patch_size: Ignored<usize>,
    /// Temporal patch size
    temporal_patch_size: Ignored<usize>,
    /// Output channels
    out_channels: Ignored<usize>,
    /// Hidden size
    hidden_size: Ignored<usize>,
}

impl<B: Backend> FinalLayer<B> {
    /// Create the final layer
    pub fn new(config: &LongCatConfig, device: &B::Device) -> Self {
        // head.head: [64, 4096] where 64 = 16 * 1 * 2 * 2 (latent_c * t * h * w patch)
        let out_dim = config.latent_channels
            * config.temporal_patch_size
            * config.patch_size
            * config.patch_size;

        Self {
            head: LinearConfig::new(config.hidden_size, out_dim)
                .with_bias(true)
                .init(device),
            // head.modulation.1: [8192, 512] = [2 * hidden_size, adaln_embed_size]
            modulation: LinearConfig::new(config.adaln_embedding_size, 2 * config.hidden_size)
                .with_bias(true)
                .init(device),
            patch_size: Ignored(config.patch_size),
            temporal_patch_size: Ignored(config.temporal_patch_size),
            out_channels: Ignored(config.latent_channels),
            hidden_size: Ignored(config.hidden_size),
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Hidden states [batch, num_patches, hidden_size]
    /// * `t_emb` - Timestep embedding [batch, adaln_embed_size]
    /// * `t_patches` - Number of temporal patches
    /// * `h_patches` - Number of height patches
    /// * `w_patches` - Number of width patches
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        t_emb: Tensor<B, 2>,
        t_patches: usize,
        h_patches: usize,
        w_patches: usize,
    ) -> Tensor<B, 5> {
        let [batch, _num_patches, _hidden] = x.dims();
        let hidden_size = self.hidden_size.0;

        // Get modulation parameters: SiLU then Linear
        let mod_out = self.modulation.forward(silu(t_emb));
        let shift = mod_out.clone().slice([0..batch, 0..hidden_size]);
        let scale = mod_out.slice([0..batch, hidden_size..2 * hidden_size]);

        // Apply modulation (no norm in head, just shift/scale)
        let x = x * (scale.unsqueeze_dim(1) + 1.0) + shift.unsqueeze_dim(1);

        // Project to output dimension
        let x = self.head.forward(x);

        // Unpatchify back to video shape
        let x = x.reshape([
            batch,
            t_patches,
            h_patches,
            w_patches,
            self.out_channels.0,
            self.temporal_patch_size.0,
            self.patch_size.0,
            self.patch_size.0,
        ]);

        // Permute back to [B, C, T, H, W]
        let x = x.permute([0, 4, 1, 5, 2, 6, 3, 7]);

        x.reshape([
            batch,
            self.out_channels.0,
            t_patches * self.temporal_patch_size.0,
            h_patches * self.patch_size.0,
            w_patches * self.patch_size.0,
        ])
    }
}
