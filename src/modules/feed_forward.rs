//! Feed-forward network for LongCat
//!
//! Uses SwiGLU activation (gated linear unit with Swish/SiLU).
//! Matches actual model: ffn.w1, ffn.w2, ffn.w3

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::silu;

use crate::LongCatConfig;

/// SwiGLU Feed-Forward Network
///
/// FFN(x) = w2 * (SiLU(w1 * x) * w3 * x)
///
/// Matches actual model naming:
/// - w1: gate projection [11008, 4096]
/// - w2: down projection [4096, 11008]
/// - w3: up projection [11008, 4096]
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    /// Gate projection: ffn.w1 [ffn_hidden, hidden]
    w1: Linear<B>,
    /// Down projection: ffn.w2 [hidden, ffn_hidden]
    w2: Linear<B>,
    /// Up projection: ffn.w3 [ffn_hidden, hidden]
    w3: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    /// Create a new SwiGLU feed-forward network
    pub fn new(config: &LongCatConfig, device: &B::Device) -> Self {
        // Actual FFN hidden size from model: 11008 (not 16384)
        // This is 2.6875 * hidden_size, a common LLaMA-style multiplier
        let ffn_hidden = config.ffn_hidden_size;

        Self {
            w1: LinearConfig::new(config.hidden_size, ffn_hidden)
                .with_bias(false)
                .init(device),
            w2: LinearConfig::new(ffn_hidden, config.hidden_size)
                .with_bias(false)
                .init(device),
            w3: LinearConfig::new(config.hidden_size, ffn_hidden)
                .with_bias(false)
                .init(device),
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, hidden_size]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // SwiGLU: w2(SiLU(w1(x)) * w3(x))
        let gate = silu(self.w1.forward(x.clone()));
        let up = self.w3.forward(x);
        let hidden = gate * up;
        self.w2.forward(hidden)
    }
}
