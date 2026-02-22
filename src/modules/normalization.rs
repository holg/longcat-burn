//! Normalization layers for LongCat
//!
//! LongCat uses RMSNorm for layer normalization and QKNorm for attention stability.

use burn::module::{Ignored, Module, Param};
use burn::prelude::*;

/// RMS Layer Normalization
///
/// Simplified version of LayerNorm that only normalizes by RMS (root mean square),
/// without subtracting the mean. Provides ~7% speedup over standard LayerNorm.
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    /// Scale parameter (gamma)
    weight: Param<Tensor<B, 1>>,
    /// Epsilon for numerical stability
    eps: Ignored<f64>,
}

impl<B: Backend> RmsNorm<B> {
    /// Create a new RMS normalization layer
    pub fn new(size: usize, eps: f64, device: &B::Device) -> Self {
        let weight = Param::from_tensor(Tensor::ones([size], device));
        Self { weight, eps: Ignored(eps) }
    }

    /// Forward pass for 3D tensor [batch, seq, hidden]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let variance = x.clone().powf_scalar(2.0).mean_dim(2);
        let x_norm = x / (variance + self.eps.0).sqrt();
        x_norm * self.weight.val().clone().unsqueeze_dims(&[0, 1])
    }

    /// Forward pass for 4D tensor [batch, seq, heads, head_dim]
    pub fn forward_4d(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let variance = x.clone().powf_scalar(2.0).mean_dim(3);
        let x_norm = x / (variance + self.eps.0).sqrt();
        x_norm * self.weight.val().clone().unsqueeze_dims(&[0, 1, 2])
    }

    /// Forward pass for 5D tensor [batch, time, height, width, channels]
    pub fn forward_5d(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let variance = x.clone().powf_scalar(2.0).mean_dim(4);
        let x_norm = x / (variance + self.eps.0).sqrt();
        x_norm * self.weight.val().clone().unsqueeze_dims(&[0, 1, 2, 3])
    }
}

/// QK Normalization for attention
///
/// Applied to query and key tensors before attention computation.
/// Helps stabilize training for long sequences in video transformers.
#[derive(Module, Debug)]
pub struct QKNorm<B: Backend> {
    /// Query normalization
    q_norm: RmsNorm<B>,
    /// Key normalization
    k_norm: RmsNorm<B>,
}

impl<B: Backend> QKNorm<B> {
    /// Create QK normalization layers
    pub fn new(head_dim: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            q_norm: RmsNorm::new(head_dim, eps, device),
            k_norm: RmsNorm::new(head_dim, eps, device),
        }
    }

    /// Normalize query and key tensors
    /// Input shape: [batch, num_heads, seq_len, head_dim]
    pub fn forward(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let q_norm = self.q_norm.forward_4d(q);
        let k_norm = self.k_norm.forward_4d(k);
        (q_norm, k_norm)
    }
}
