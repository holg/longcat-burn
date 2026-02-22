//! LongCat model configuration

use burn::config::Config;

/// Configuration for the LongCat video generation model
#[derive(Config, Debug)]
pub struct LongCatConfig {
    /// Number of transformer layers (default: 48)
    #[config(default = 48)]
    pub num_layers: usize,

    /// Hidden dimension size (default: 4096)
    #[config(default = 4096)]
    pub hidden_size: usize,

    /// Feed-forward network hidden size (default: 11008, matching LLaMA-style)
    #[config(default = 11008)]
    pub ffn_hidden_size: usize,

    /// Number of attention heads (default: 32)
    #[config(default = 32)]
    pub num_attention_heads: usize,

    /// Attention head dimension (default: 128)
    #[config(default = 128)]
    pub head_dim: usize,

    /// AdaLN embedding size for timestep modulation (default: 512)
    #[config(default = 512)]
    pub adaln_embedding_size: usize,

    /// Number of latent channels from VAE (default: 16)
    #[config(default = 16)]
    pub latent_channels: usize,

    /// Patch size for spatial dimensions (default: 2)
    #[config(default = 2)]
    pub patch_size: usize,

    /// Temporal patch size (default: 1)
    #[config(default = 1)]
    pub temporal_patch_size: usize,

    /// Text encoder hidden dimension (UMT5-XXL: 4096)
    #[config(default = 4096)]
    pub text_hidden_size: usize,

    /// Maximum sequence length for text (default: 512)
    #[config(default = 512)]
    pub max_text_seq_len: usize,

    /// Layer norm epsilon (default: 1e-6)
    #[config(default = 1e-6)]
    pub norm_eps: f64,

    /// Whether to use QK normalization (default: true)
    #[config(default = true)]
    pub qk_norm: bool,

    /// RoPE theta base (default: 10000.0)
    #[config(default = 10000.0)]
    pub rope_theta: f64,
}

impl Default for LongCatConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl LongCatConfig {
    /// Get the total number of parameters (approximate)
    pub fn num_params(&self) -> usize {
        // Rough estimate: 13.6B for default config
        let embed_params = self.latent_channels * self.patch_size * self.patch_size * self.hidden_size;
        let layer_params = self.num_layers * (
            // Self-attention
            4 * self.hidden_size * self.hidden_size +
            // Cross-attention
            4 * self.hidden_size * self.hidden_size +
            // FFN
            3 * self.hidden_size * self.ffn_hidden_size +
            // Norms and modulation
            10 * self.hidden_size
        );
        let final_params = self.hidden_size * self.latent_channels * self.patch_size * self.patch_size;

        embed_params + layer_params + final_params
    }
}

/// Configuration for video generation
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    /// Number of frames to generate
    pub num_frames: usize,
    /// Video height in pixels
    pub height: usize,
    /// Video width in pixels
    pub width: usize,
    /// Frames per second
    pub fps: usize,
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Classifier-free guidance scale
    pub guidance_scale: f32,
    /// Random seed (optional)
    pub seed: Option<u64>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            num_frames: 81, // ~5 seconds at 15fps
            height: 480,
            width: 832,
            fps: 15,
            num_inference_steps: 50,
            guidance_scale: 5.0,
            seed: None,
        }
    }
}

impl GenerateConfig {
    /// Configuration for 480p @ 15fps (fast)
    pub fn fast_480p() -> Self {
        Self {
            num_frames: 41,
            height: 480,
            width: 832,
            fps: 15,
            num_inference_steps: 25,
            guidance_scale: 5.0,
            seed: None,
        }
    }

    /// Configuration for 720p @ 30fps (quality)
    pub fn quality_720p() -> Self {
        Self {
            num_frames: 81,
            height: 720,
            width: 1280,
            fps: 30,
            num_inference_steps: 50,
            guidance_scale: 5.0,
            seed: None,
        }
    }
}
