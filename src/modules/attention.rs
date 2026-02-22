//! Attention modules for LongCat
//!
//! Implements 3D self-attention for spatial-temporal modeling and
//! cross-attention for text conditioning.
//!
//! Supports memory-efficient sliced attention for large sequences.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;

use super::memory::get_attention_slice_size;
use super::rope::Rope3DFreqs;
use crate::LongCatConfig;

/// 3D Self-Attention
///
/// Attends over spatial-temporal tokens with 3D RoPE positional encoding.
/// Matches the actual LongCat/Kijai safetensors structure.
#[derive(Module, Debug)]
pub struct SelfAttention3D<B: Backend> {
    /// Query projection
    q: Linear<B>,
    /// Key projection
    k: Linear<B>,
    /// Value projection
    v: Linear<B>,
    /// Output projection
    o: Linear<B>,
    /// Query normalization (RMSNorm on head_dim)
    norm_q: RmsNorm<B>,
    /// Key normalization (RMSNorm on head_dim)
    norm_k: RmsNorm<B>,
    /// Number of attention heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Scale factor for attention
    scale: f32,
}

impl<B: Backend> SelfAttention3D<B> {
    /// Create a new 3D self-attention module
    pub fn new(config: &LongCatConfig, device: &B::Device) -> Self {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = config.head_dim;

        Self {
            q: LinearConfig::new(hidden_size, num_heads * head_dim)
                .with_bias(true)
                .init(device),
            k: LinearConfig::new(hidden_size, num_heads * head_dim)
                .with_bias(true)
                .init(device),
            v: LinearConfig::new(hidden_size, num_heads * head_dim)
                .with_bias(true)
                .init(device),
            o: LinearConfig::new(num_heads * head_dim, hidden_size)
                .with_bias(true)
                .init(device),
            norm_q: RmsNormConfig::new(head_dim)
                .with_epsilon(config.norm_eps)
                .init(device),
            norm_k: RmsNormConfig::new(head_dim)
                .with_epsilon(config.norm_eps)
                .init(device),
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// Forward pass with optional RoPE
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, hidden_size]
    /// * `rope_freqs` - Optional precomputed RoPE frequencies
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope_freqs: Option<&Rope3DFreqs<B>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();

        // Project to Q, K, V
        let q = self.q.forward(x.clone());
        let k = self.k.forward(x.clone());
        let v = self.v.forward(x);

        // Reshape to [batch, num_heads, seq_len, head_dim]
        let q = q.reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k.reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v.reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        // Apply RoPE if provided
        let (q, k) = if let Some(freqs) = rope_freqs {
            (
                self.apply_rope(q, &freqs.cos, &freqs.sin),
                self.apply_rope(k, &freqs.cos, &freqs.sin),
            )
        } else {
            (q, k)
        };

        // Apply QK normalization per head - RMSNorm on head_dim
        // Reshape to [batch * num_heads * seq_len, head_dim] for norm
        let q = q.permute([0, 2, 1, 3]).reshape([batch * seq_len, self.num_heads, self.head_dim]);
        let k = k.permute([0, 2, 1, 3]).reshape([batch * seq_len, self.num_heads, self.head_dim]);

        // Apply norm per head
        let q = self.norm_q.forward(q);
        let k = self.norm_k.forward(k);

        // Reshape back to [batch, num_heads, seq_len, head_dim]
        let q = q.reshape([batch, seq_len, self.num_heads, self.head_dim]).swap_dims(1, 2);
        let k = k.reshape([batch, seq_len, self.num_heads, self.head_dim]).swap_dims(1, 2);

        // Check if we should use sliced attention
        let slice_size = get_attention_slice_size();
        let out = if slice_size > 0 && seq_len > slice_size {
            self.sliced_attention(q, k, v, slice_size)
        } else {
            self.full_attention(q, k, v)
        };

        // Reshape back to [batch, seq_len, hidden_size]
        let out = out.swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);

        self.o.forward(out)
    }

    /// Full attention computation (original, memory-intensive)
    fn full_attention(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        // Compute attention scores
        let attn = q.matmul(k.transpose()) * self.scale;
        let attn = softmax(attn, 3);

        // Apply attention to values
        attn.matmul(v)
    }

    /// Sliced attention computation (memory-efficient)
    ///
    /// Computes attention in chunks to reduce peak memory usage.
    /// For self-attention, we slice along the query dimension.
    fn sliced_attention(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        slice_size: usize,
    ) -> Tensor<B, 4> {
        let [batch, num_heads, seq_len, head_dim] = q.dims();

        // Number of slices needed
        let num_slices = (seq_len + slice_size - 1) / slice_size;

        // Process each slice
        let mut outputs: Vec<Tensor<B, 4>> = Vec::with_capacity(num_slices);

        for i in 0..num_slices {
            let start = i * slice_size;
            let end = ((i + 1) * slice_size).min(seq_len);

            // Get query slice [batch, num_heads, slice_len, head_dim]
            let q_slice = q.clone().slice([
                0..batch,
                0..num_heads,
                start..end,
                0..head_dim,
            ]);

            // Compute attention for this slice against all keys
            // attn_slice: [batch, num_heads, slice_len, seq_len]
            let attn_slice = q_slice.matmul(k.clone().transpose()) * self.scale;
            let attn_slice = softmax(attn_slice, 3);

            // Apply to values: [batch, num_heads, slice_len, head_dim]
            let out_slice = attn_slice.matmul(v.clone());
            outputs.push(out_slice);
        }

        // Concatenate all slices along sequence dimension
        if outputs.len() == 1 {
            outputs.pop().unwrap()
        } else {
            Tensor::cat(outputs, 2)
        }
    }

    /// Apply rotary position embedding
    fn apply_rope(
        &self,
        x: Tensor<B, 4>,
        cos: &Tensor<B, 2>,
        sin: &Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        let [batch, heads, seq, dim] = x.dims();
        let half_dim = dim / 2;

        // Expand cos/sin to match x shape
        let cos = cos.clone().unsqueeze_dims(&[0, 1]).repeat(&[batch, heads, 1, 1]);
        let sin = sin.clone().unsqueeze_dims(&[0, 1]).repeat(&[batch, heads, 1, 1]);

        // Split into two halves
        let x1 = x.clone().slice([0..batch, 0..heads, 0..seq, 0..half_dim]);
        let x2 = x.slice([0..batch, 0..heads, 0..seq, half_dim..dim]);

        // Apply rotation
        let x1_rot = x1.clone() * cos.clone() - x2.clone() * sin.clone();
        let x2_rot = x1 * sin + x2 * cos;

        Tensor::cat(vec![x1_rot, x2_rot], 3)
    }
}

/// Cross-Attention for text conditioning
///
/// Queries attend to text encoder outputs (keys/values).
/// Matches the actual LongCat/Kijai safetensors structure.
#[derive(Module, Debug)]
pub struct CrossAttention<B: Backend> {
    /// Query projection (from video tokens)
    q: Linear<B>,
    /// Key projection (from text tokens)
    k: Linear<B>,
    /// Value projection (from text tokens)
    v: Linear<B>,
    /// Output projection
    o: Linear<B>,
    /// Query normalization (RMSNorm on head_dim)
    norm_q: RmsNorm<B>,
    /// Key normalization (RMSNorm on head_dim)
    norm_k: RmsNorm<B>,
    /// Number of attention heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Scale factor
    scale: f32,
}

impl<B: Backend> CrossAttention<B> {
    /// Create a new cross-attention module
    pub fn new(config: &LongCatConfig, device: &B::Device) -> Self {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = config.head_dim;

        Self {
            q: LinearConfig::new(hidden_size, num_heads * head_dim)
                .with_bias(true)
                .init(device),
            k: LinearConfig::new(config.text_hidden_size, num_heads * head_dim)
                .with_bias(true)
                .init(device),
            v: LinearConfig::new(config.text_hidden_size, num_heads * head_dim)
                .with_bias(true)
                .init(device),
            o: LinearConfig::new(num_heads * head_dim, hidden_size)
                .with_bias(true)
                .init(device),
            norm_q: RmsNormConfig::new(head_dim)
                .with_epsilon(config.norm_eps)
                .init(device),
            norm_k: RmsNormConfig::new(head_dim)
                .with_epsilon(config.norm_eps)
                .init(device),
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Video tokens [batch, video_seq_len, hidden_size]
    /// * `context` - Text embeddings [batch, text_seq_len, text_hidden_size]
    pub fn forward(&self, x: Tensor<B, 3>, context: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, video_seq, _] = x.dims();
        let [_, text_seq, _] = context.dims();

        // Project queries from video, keys/values from text
        let q = self.q.forward(x);
        let k = self.k.forward(context.clone());
        let v = self.v.forward(context);

        // Reshape to multi-head format [batch, num_heads, seq, head_dim]
        let q = q.reshape([batch, video_seq, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k.reshape([batch, text_seq, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v.reshape([batch, text_seq, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        // Apply QK normalization per head
        let q = q.permute([0, 2, 1, 3]).reshape([batch * video_seq, self.num_heads, self.head_dim]);
        let k = k.permute([0, 2, 1, 3]).reshape([batch * text_seq, self.num_heads, self.head_dim]);

        let q = self.norm_q.forward(q);
        let k = self.norm_k.forward(k);

        let q = q.reshape([batch, video_seq, self.num_heads, self.head_dim]).swap_dims(1, 2);
        let k = k.reshape([batch, text_seq, self.num_heads, self.head_dim]).swap_dims(1, 2);

        // Check if we should use sliced attention
        let slice_size = get_attention_slice_size();
        let out = if slice_size > 0 && video_seq > slice_size {
            self.sliced_attention(q, k, v, slice_size)
        } else {
            self.full_attention(q, k, v)
        };

        // Reshape back
        let out = out.swap_dims(1, 2)
            .reshape([batch, video_seq, self.num_heads * self.head_dim]);

        self.o.forward(out)
    }

    /// Full attention computation
    fn full_attention(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let attn = q.matmul(k.transpose()) * self.scale;
        let attn = softmax(attn, 3);
        attn.matmul(v)
    }

    /// Sliced attention for cross-attention
    ///
    /// Slices along query (video) dimension since text sequence is typically short.
    fn sliced_attention(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        slice_size: usize,
    ) -> Tensor<B, 4> {
        let [batch, num_heads, video_seq, head_dim] = q.dims();

        let num_slices = (video_seq + slice_size - 1) / slice_size;
        let mut outputs: Vec<Tensor<B, 4>> = Vec::with_capacity(num_slices);

        for i in 0..num_slices {
            let start = i * slice_size;
            let end = ((i + 1) * slice_size).min(video_seq);

            // Get query slice
            let q_slice = q.clone().slice([
                0..batch,
                0..num_heads,
                start..end,
                0..head_dim,
            ]);

            // Compute attention against all keys (text tokens)
            let attn_slice = q_slice.matmul(k.clone().transpose()) * self.scale;
            let attn_slice = softmax(attn_slice, 3);

            // Apply to values
            let out_slice = attn_slice.matmul(v.clone());
            outputs.push(out_slice);
        }

        if outputs.len() == 1 {
            outputs.pop().unwrap()
        } else {
            Tensor::cat(outputs, 2)
        }
    }
}
