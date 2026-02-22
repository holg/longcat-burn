//! AdaLN-Zero Modulation for LongCat
//!
//! Provides timestep-dependent layer-wise control through adaptive layer normalization.
//! Zero initialization ensures stable training.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::silu;

use crate::LongCatConfig;

/// AdaLN-Zero Modulation
///
/// Generates scale and shift parameters for layer normalization based on timestep embedding.
/// Each transformer block has its own modulation MLP.
#[derive(Module, Debug)]
pub struct AdaLNModulation<B: Backend> {
    /// Linear projection from timestep embedding
    linear: Linear<B>,
    /// Output dimension (6 * hidden_size for scale, shift, gate for both norm layers)
    out_dim: usize,
}

impl<B: Backend> AdaLNModulation<B> {
    /// Create a new AdaLN modulation layer
    ///
    /// # Arguments
    /// * `embed_dim` - Timestep embedding dimension (adaln_embedding_size)
    /// * `hidden_size` - Model hidden dimension
    pub fn new(embed_dim: usize, hidden_size: usize, device: &B::Device) -> Self {
        // Output: 6 modulation parameters per position
        // (shift1, scale1, gate1, shift2, scale2, gate2)
        let out_dim = 6 * hidden_size;
        let linear = LinearConfig::new(embed_dim, out_dim)
            .with_bias(true)
            .init(device);

        Self { linear, out_dim }
    }

    /// Compute modulation parameters from timestep embedding
    ///
    /// # Arguments
    /// * `t_emb` - Timestep embedding [batch, embed_dim]
    ///
    /// # Returns
    /// Tuple of (shift1, scale1, gate1, shift2, scale2, gate2)
    /// Each has shape [batch, hidden_size]
    pub fn forward(
        &self,
        t_emb: Tensor<B, 2>,
    ) -> (
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
    ) {
        let modulation = self.linear.forward(silu(t_emb));
        let hidden_size = self.out_dim / 6;
        let [batch, _] = modulation.dims();

        // Split into 6 parts
        let shift1 = modulation.clone().slice([0..batch, 0..hidden_size]);
        let scale1 = modulation.clone().slice([0..batch, hidden_size..2 * hidden_size]);
        let gate1 = modulation.clone().slice([0..batch, 2 * hidden_size..3 * hidden_size]);
        let shift2 = modulation.clone().slice([0..batch, 3 * hidden_size..4 * hidden_size]);
        let scale2 = modulation.clone().slice([0..batch, 4 * hidden_size..5 * hidden_size]);
        let gate2 = modulation.slice([0..batch, 5 * hidden_size..6 * hidden_size]);

        (shift1, scale1, gate1, shift2, scale2, gate2)
    }
}

/// Timestep Embedder
///
/// Matches actual model: time_embedding.mlp.0/2
/// Converts scalar timesteps to embeddings using sinusoidal encoding + MLP.
#[derive(Module, Debug)]
pub struct TimestepEmbedder<B: Backend> {
    /// MLP module containing the two linear layers
    mlp: TimestepMLP<B>,
    /// Frequency embedding size (256 = adaln_embedding_size / 2)
    freq_embed_size: usize,
}

/// MLP for timestep embedding: time_embedding.mlp
#[derive(Module, Debug)]
pub struct TimestepMLP<B: Backend> {
    /// First linear layer: time_embedding.mlp.0 [512, 256]
    linear1: Linear<B>,
    /// Second linear layer: time_embedding.mlp.2 [512, 512]
    linear2: Linear<B>,
}

impl<B: Backend> TimestepEmbedder<B> {
    /// Create a new timestep embedder
    pub fn new(config: &LongCatConfig, device: &B::Device) -> Self {
        // Frequency embedding size is half of adaln_embedding_size
        // time_embedding.mlp.0: [512, 256] -> input is 256 (freq embed), output 512
        // time_embedding.mlp.2: [512, 512] -> 512 -> 512
        let freq_embed_size = config.adaln_embedding_size / 2; // 256

        Self {
            mlp: TimestepMLP {
                linear1: LinearConfig::new(freq_embed_size, config.adaln_embedding_size)
                    .with_bias(true)
                    .init(device),
                linear2: LinearConfig::new(config.adaln_embedding_size, config.adaln_embedding_size)
                    .with_bias(true)
                    .init(device),
            },
            freq_embed_size,
        }
    }

    /// Convert timesteps to embeddings
    ///
    /// # Arguments
    /// * `t` - Timesteps [batch] in range [0, 1]
    pub fn forward(&self, t: Tensor<B, 1>) -> Tensor<B, 2> {
        let t_freq = self.timestep_embedding(t, self.freq_embed_size);
        let t_emb = self.mlp.linear1.forward(t_freq);
        let t_emb = silu(t_emb);
        self.mlp.linear2.forward(t_emb)
    }

    /// Sinusoidal timestep embedding
    fn timestep_embedding(&self, t: Tensor<B, 1>, dim: usize) -> Tensor<B, 2> {
        let device = t.device();
        let half_dim = dim / 2;

        // Compute frequencies
        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| (-((i as f32) / half_dim as f32) * 10000.0_f32.ln()).exp())
            .collect();
        let freqs = Tensor::<B, 1>::from_floats(freqs.as_slice(), &device);

        // [batch, half_dim]
        let args = t.unsqueeze_dim(1) * freqs.unsqueeze_dim(0);

        // Concatenate sin and cos
        Tensor::cat(vec![args.clone().cos(), args.sin()], 1)
    }
}
