//! WAN 2.1 Video VAE for LongCat
//!
//! 3D Causal Variational Autoencoder for video encoding/decoding.
//! Architecture matches the actual WAN VAE safetensors structure.

use burn::module::{Ignored, Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig, Conv3d, Conv3dConfig};
use burn::nn::PaddingConfig2d;
use burn::nn::PaddingConfig3d;
use burn::prelude::*;
use std::path::PathBuf;

use burn::store::{BurnpackStore, ModuleStore, SafetensorsStore, PyTorchToBurnAdapter};

/// WAN 2.1 VAE Configuration
#[derive(Debug, Clone)]
pub struct WanVaeConfig {
    pub latent_channels: usize,
    pub norm_eps: f64,
}

impl Default for WanVaeConfig {
    fn default() -> Self {
        Self {
            latent_channels: 16,
            norm_eps: 1e-6,
        }
    }
}

/// SiLU activation for 5D tensors
fn silu_5d<B: Backend>(x: Tensor<B, 5>) -> Tensor<B, 5> {
    x.clone() * burn::tensor::activation::sigmoid(x)
}

/// 3D Group Normalization using gamma (no beta)
/// Matches: residual.0.gamma, residual.3.gamma pattern
#[derive(Module, Debug)]
pub struct CausalNorm3D<B: Backend> {
    gamma: Param<Tensor<B, 4>>, // [channels, 1, 1, 1]
    num_channels: Ignored<usize>,
    eps: Ignored<f64>,
}

impl<B: Backend> CausalNorm3D<B> {
    pub fn new(num_channels: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            gamma: Param::from_tensor(Tensor::ones([num_channels, 1, 1, 1], device)),
            num_channels: Ignored(num_channels),
            eps: Ignored(eps),
        }
    }

    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let [batch, channels, time, height, width] = x.dims();

        // Group norm with 32 groups
        let num_groups = 32.min(channels);
        let group_size = channels / num_groups;

        let x_reshaped = x.reshape([batch, num_groups, group_size * time * height * width]);
        let mean = x_reshaped.clone().mean_dim(2);
        let var = x_reshaped.clone().var(2);
        let x_norm = (x_reshaped - mean) / (var + self.eps.0).sqrt();
        let x_norm = x_norm.reshape([batch, channels, time, height, width]);

        // Apply gamma (expand to [1, channels, 1, 1, 1])
        let gamma = self.gamma.val().clone().unsqueeze_dim::<5>(0);
        x_norm * gamma
    }
}

/// 2D Spatial Normalization (for attention)
#[derive(Module, Debug)]
pub struct SpatialNorm2D<B: Backend> {
    gamma: Param<Tensor<B, 3>>, // [channels, 1, 1]
    num_channels: Ignored<usize>,
    eps: Ignored<f64>,
}

impl<B: Backend> SpatialNorm2D<B> {
    pub fn new(num_channels: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            gamma: Param::from_tensor(Tensor::ones([num_channels, 1, 1], device)),
            num_channels: Ignored(num_channels),
            eps: Ignored(eps),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = x.dims();
        let num_groups = 32.min(channels);
        let group_size = channels / num_groups;

        let x_reshaped = x.reshape([batch, num_groups, group_size * height * width]);
        let mean = x_reshaped.clone().mean_dim(2);
        let var = x_reshaped.clone().var(2);
        let x_norm = (x_reshaped - mean) / (var + self.eps.0).sqrt();
        let x_norm = x_norm.reshape([batch, channels, height, width]);

        let gamma = self.gamma.val().clone().unsqueeze_dim::<4>(0);
        x_norm * gamma
    }
}

/// Inner residual block structure matching safetensors naming
/// residual.0.gamma, residual.2.weight/bias, residual.3.gamma, residual.6.weight/bias
#[derive(Module, Debug)]
struct ResidualInner<B: Backend> {
    norm1: CausalNorm3D<B>,
    conv1: Conv3d<B>,
    norm2: CausalNorm3D<B>,
    conv2: Conv3d<B>,
}

/// Residual block with optional shortcut
#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    residual: ResidualInner<B>,
    shortcut: Option<Conv3d<B>>,
}

impl<B: Backend> ResidualBlock<B> {
    pub fn new(in_channels: usize, out_channels: usize, eps: f64, device: &B::Device) -> Self {
        let shortcut = if in_channels != out_channels {
            Some(
                Conv3dConfig::new([in_channels, out_channels], [1, 1, 1])
                    .init(device),
            )
        } else {
            None
        };

        Self {
            residual: ResidualInner {
                norm1: CausalNorm3D::new(in_channels, eps, device),
                conv1: Conv3dConfig::new([in_channels, out_channels], [3, 3, 3])
                    .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
                    .init(device),
                norm2: CausalNorm3D::new(out_channels, eps, device),
                conv2: Conv3dConfig::new([out_channels, out_channels], [3, 3, 3])
                    .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
                    .init(device),
            },
            shortcut,
        }
    }

    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let residual = match &self.shortcut {
            Some(conv) => conv.forward(x.clone()),
            None => x.clone(),
        };

        let h = self.residual.norm1.forward(x);
        let h = silu_5d(h);
        let h = self.residual.conv1.forward(h);
        let h = self.residual.norm2.forward(h);
        let h = silu_5d(h);
        let h = self.residual.conv2.forward(h);

        h + residual
    }
}

/// Inner resample structure
#[derive(Module, Debug)]
struct ResampleInner<B: Backend> {
    conv: Conv2d<B>,
}

/// Spatial resample block
#[derive(Module, Debug)]
pub struct SpatialResample<B: Backend> {
    resample: ResampleInner<B>,
    is_upsample: Ignored<bool>,
}

impl<B: Backend> SpatialResample<B> {
    pub fn new_downsample(channels: usize, device: &B::Device) -> Self {
        Self {
            resample: ResampleInner {
                conv: Conv2dConfig::new([channels, channels], [3, 3])
                    .with_stride([2, 2])
                    .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                    .init(device),
            },
            is_upsample: Ignored(false),
        }
    }

    pub fn new_upsample(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        Self {
            resample: ResampleInner {
                conv: Conv2dConfig::new([in_channels, out_channels], [3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                    .init(device),
            },
            is_upsample: Ignored(true),
        }
    }

    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let [batch, channels, time, height, width] = x.dims();

        if self.is_upsample.0 {
            // Upsample spatially first (nearest neighbor)
            let x = x.reshape([batch, channels, time, height, 1, width, 1])
                .repeat(&[1, 1, 1, 1, 2, 1, 2])
                .reshape([batch, channels, time, height * 2, width * 2]);
            let [batch, channels, time, height, width] = x.dims();
            let x = x.swap_dims(1, 2).reshape([batch * time, channels, height, width]);
            let x = self.resample.conv.forward(x);
            let [_, c, h, w] = x.dims();
            x.reshape([batch, time, c, h, w]).swap_dims(1, 2)
        } else {
            // Downsample
            let x = x.swap_dims(1, 2).reshape([batch * time, channels, height, width]);
            let x = self.resample.conv.forward(x);
            let [_, c, h, w] = x.dims();
            x.reshape([batch, time, c, h, w]).swap_dims(1, 2)
        }
    }
}

/// Temporal convolution for time resampling
#[derive(Module, Debug)]
pub struct TemporalConv<B: Backend> {
    time_conv: Conv3d<B>,
    is_upsample: Ignored<bool>,
}

impl<B: Backend> TemporalConv<B> {
    pub fn new_downsample(channels: usize, device: &B::Device) -> Self {
        Self {
            time_conv: Conv3dConfig::new([channels, channels], [3, 1, 1])
                .with_stride([2, 1, 1])
                .with_padding(PaddingConfig3d::Explicit(1, 0, 0))
                .init(device),
            is_upsample: Ignored(false),
        }
    }

    pub fn new_upsample(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        Self {
            time_conv: Conv3dConfig::new([in_channels, out_channels], [3, 1, 1])
                .with_padding(PaddingConfig3d::Explicit(1, 0, 0))
                .init(device),
            is_upsample: Ignored(true),
        }
    }

    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        if self.is_upsample.0 {
            let [batch, channels, time, height, width] = x.dims();
            let x = x.reshape([batch, channels, time, 1, height, width])
                .repeat(&[1, 1, 1, 2, 1, 1])
                .reshape([batch, channels, time * 2, height, width]);
            self.time_conv.forward(x)
        } else {
            self.time_conv.forward(x)
        }
    }
}

/// Spatio-temporal resample block
#[derive(Module, Debug)]
pub struct SpatioTemporalResample<B: Backend> {
    resample: SpatialResample<B>,
    time_conv: TemporalConv<B>,
}

impl<B: Backend> SpatioTemporalResample<B> {
    pub fn new_downsample(channels: usize, device: &B::Device) -> Self {
        Self {
            resample: SpatialResample::new_downsample(channels, device),
            time_conv: TemporalConv::new_downsample(channels, device),
        }
    }

    pub fn new_upsample(in_channels: usize, out_spatial: usize, out_temporal: usize, device: &B::Device) -> Self {
        Self {
            resample: SpatialResample::new_upsample(in_channels, out_spatial, device),
            time_conv: TemporalConv::new_upsample(in_channels, out_temporal, device),
        }
    }

    pub fn forward_down(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let x = self.resample.forward(x);
        self.time_conv.forward(x)
    }

    pub fn forward_up(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let x = self.time_conv.forward(x);
        self.resample.forward(x)
    }
}

/// Middle attention block
#[derive(Module, Debug)]
pub struct MiddleAttention<B: Backend> {
    norm: SpatialNorm2D<B>,
    to_qkv: Conv2d<B>,
    proj: Conv2d<B>,
    num_heads: Ignored<usize>,
}

impl<B: Backend> MiddleAttention<B> {
    pub fn new(channels: usize, num_heads: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            norm: SpatialNorm2D::new(channels, eps, device),
            to_qkv: Conv2dConfig::new([channels, channels * 3], [1, 1]).init(device),
            proj: Conv2dConfig::new([channels, channels], [1, 1]).init(device),
            num_heads: Ignored(num_heads),
        }
    }

    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let [batch, channels, time, height, width] = x.dims();

        let x_2d = x.clone().swap_dims(1, 2).reshape([batch * time, channels, height, width]);
        let x_norm = self.norm.forward(x_2d);

        let qkv = self.to_qkv.forward(x_norm);
        let [bt, _c3, h, w] = qkv.dims();
        let head_dim = channels / self.num_heads.0;

        let qkv = qkv.reshape([bt, 3, self.num_heads.0, head_dim, h * w]);
        let q = qkv.clone().slice([0..bt, 0..1, 0..self.num_heads.0, 0..head_dim, 0..h*w]).squeeze::<4>();
        let k = qkv.clone().slice([0..bt, 1..2, 0..self.num_heads.0, 0..head_dim, 0..h*w]).squeeze::<4>();
        let v = qkv.slice([0..bt, 2..3, 0..self.num_heads.0, 0..head_dim, 0..h*w]).squeeze::<4>();

        let scale = (head_dim as f64).sqrt();
        let attn = q.matmul(k.swap_dims(2, 3)) / scale;
        let attn = burn::tensor::activation::softmax(attn, 3);

        let out = attn.matmul(v);
        let out = out.reshape([bt, channels, h, w]);

        let out = self.proj.forward(out);
        let out = out.reshape([batch, time, channels, height, width]).swap_dims(1, 2);

        x + out
    }
}

/// Head block with norm and conv
#[derive(Module, Debug)]
struct HeadInner<B: Backend> {
    norm: CausalNorm3D<B>,
    conv: Conv3d<B>,
}

/// Encoder head
#[derive(Module, Debug)]
pub struct EncoderHead<B: Backend> {
    head: HeadInner<B>,
}

impl<B: Backend> EncoderHead<B> {
    pub fn new(in_channels: usize, out_channels: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            head: HeadInner {
                norm: CausalNorm3D::new(in_channels, eps, device),
                conv: Conv3dConfig::new([in_channels, out_channels], [3, 3, 3])
                    .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
                    .init(device),
            },
        }
    }

    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let h = self.head.norm.forward(x);
        let h = silu_5d(h);
        self.head.conv.forward(h)
    }
}

/// Decoder head
#[derive(Module, Debug)]
pub struct DecoderHead<B: Backend> {
    head: HeadInner<B>,
}

impl<B: Backend> DecoderHead<B> {
    pub fn new(in_channels: usize, out_channels: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            head: HeadInner {
                norm: CausalNorm3D::new(in_channels, eps, device),
                conv: Conv3dConfig::new([in_channels, out_channels], [3, 3, 3])
                    .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
                    .init(device),
            },
        }
    }

    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let h = self.head.norm.forward(x);
        let h = silu_5d(h);
        self.head.conv.forward(h)
    }
}

/// WAN VAE Encoder with flat block vectors
#[derive(Module, Debug)]
pub struct WanVaeEncoder<B: Backend> {
    conv1: Conv3d<B>,
    // downsamples - using individual named blocks to match safetensors structure
    downsamples_0: ResidualBlock<B>,
    downsamples_1: ResidualBlock<B>,
    downsamples_2: SpatialResample<B>,
    downsamples_3: ResidualBlock<B>,
    downsamples_4: ResidualBlock<B>,
    downsamples_5: SpatioTemporalResample<B>,
    downsamples_6: ResidualBlock<B>,
    downsamples_7: ResidualBlock<B>,
    downsamples_8: SpatioTemporalResample<B>,
    downsamples_9: ResidualBlock<B>,
    downsamples_10: ResidualBlock<B>,
    // middle
    middle_0: ResidualBlock<B>,
    middle_1: MiddleAttention<B>,
    middle_2: ResidualBlock<B>,
    // head
    head: EncoderHead<B>,
}

impl<B: Backend> WanVaeEncoder<B> {
    pub fn new(config: &WanVaeConfig, device: &B::Device) -> Self {
        let eps = config.norm_eps;

        Self {
            conv1: Conv3dConfig::new([3, 96], [3, 3, 3])
                .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
                .init(device),
            downsamples_0: ResidualBlock::new(96, 96, eps, device),
            downsamples_1: ResidualBlock::new(96, 96, eps, device),
            downsamples_2: SpatialResample::new_downsample(96, device),
            downsamples_3: ResidualBlock::new(96, 192, eps, device),
            downsamples_4: ResidualBlock::new(192, 192, eps, device),
            downsamples_5: SpatioTemporalResample::new_downsample(192, device),
            downsamples_6: ResidualBlock::new(192, 384, eps, device),
            downsamples_7: ResidualBlock::new(384, 384, eps, device),
            downsamples_8: SpatioTemporalResample::new_downsample(384, device),
            downsamples_9: ResidualBlock::new(384, 384, eps, device),
            downsamples_10: ResidualBlock::new(384, 384, eps, device),
            middle_0: ResidualBlock::new(384, 384, eps, device),
            middle_1: MiddleAttention::new(384, 1, eps, device),
            middle_2: ResidualBlock::new(384, 384, eps, device),
            head: EncoderHead::new(384, 32, eps, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let mut h = self.conv1.forward(x);
        h = self.downsamples_0.forward(h);
        h = self.downsamples_1.forward(h);
        h = self.downsamples_2.forward(h);
        h = self.downsamples_3.forward(h);
        h = self.downsamples_4.forward(h);
        h = self.downsamples_5.forward_down(h);
        h = self.downsamples_6.forward(h);
        h = self.downsamples_7.forward(h);
        h = self.downsamples_8.forward_down(h);
        h = self.downsamples_9.forward(h);
        h = self.downsamples_10.forward(h);
        h = self.middle_0.forward(h);
        h = self.middle_1.forward(h);
        h = self.middle_2.forward(h);
        self.head.forward(h)
    }
}

/// WAN VAE Decoder
#[derive(Module, Debug)]
pub struct WanVaeDecoder<B: Backend> {
    conv1: Conv3d<B>,
    // middle
    middle_0: ResidualBlock<B>,
    middle_1: MiddleAttention<B>,
    middle_2: ResidualBlock<B>,
    // upsamples
    upsamples_0: ResidualBlock<B>,
    upsamples_1: ResidualBlock<B>,
    upsamples_2: ResidualBlock<B>,
    upsamples_3: SpatioTemporalResample<B>,
    upsamples_4: ResidualBlock<B>,
    upsamples_5: ResidualBlock<B>,
    upsamples_6: ResidualBlock<B>,
    upsamples_7: SpatioTemporalResample<B>,
    upsamples_8: ResidualBlock<B>,
    upsamples_9: ResidualBlock<B>,
    upsamples_10: ResidualBlock<B>,
    upsamples_11: SpatialResample<B>,
    upsamples_12: ResidualBlock<B>,
    upsamples_13: ResidualBlock<B>,
    upsamples_14: ResidualBlock<B>,
    // head
    head: DecoderHead<B>,
}

impl<B: Backend> WanVaeDecoder<B> {
    pub fn new(config: &WanVaeConfig, device: &B::Device) -> Self {
        let eps = config.norm_eps;

        Self {
            conv1: Conv3dConfig::new([16, 384], [3, 3, 3])
                .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
                .init(device),
            middle_0: ResidualBlock::new(384, 384, eps, device),
            middle_1: MiddleAttention::new(384, 1, eps, device),
            middle_2: ResidualBlock::new(384, 384, eps, device),
            upsamples_0: ResidualBlock::new(384, 384, eps, device),
            upsamples_1: ResidualBlock::new(384, 384, eps, device),
            upsamples_2: ResidualBlock::new(384, 384, eps, device),
            upsamples_3: SpatioTemporalResample::new_upsample(384, 192, 768, device),
            upsamples_4: ResidualBlock::new(192, 384, eps, device),
            upsamples_5: ResidualBlock::new(384, 384, eps, device),
            upsamples_6: ResidualBlock::new(384, 384, eps, device),
            upsamples_7: SpatioTemporalResample::new_upsample(384, 192, 768, device),
            upsamples_8: ResidualBlock::new(192, 192, eps, device),
            upsamples_9: ResidualBlock::new(192, 192, eps, device),
            upsamples_10: ResidualBlock::new(192, 192, eps, device),
            upsamples_11: SpatialResample::new_upsample(192, 96, device),
            upsamples_12: ResidualBlock::new(96, 96, eps, device),
            upsamples_13: ResidualBlock::new(96, 96, eps, device),
            upsamples_14: ResidualBlock::new(96, 96, eps, device),
            head: DecoderHead::new(96, 3, eps, device),
        }
    }

    pub fn forward(&self, z: Tensor<B, 5>) -> Tensor<B, 5> {
        let mut h = self.conv1.forward(z);
        h = self.middle_0.forward(h);
        h = self.middle_1.forward(h);
        h = self.middle_2.forward(h);
        h = self.upsamples_0.forward(h);
        h = self.upsamples_1.forward(h);
        h = self.upsamples_2.forward(h);
        h = self.upsamples_3.forward_up(h);
        h = self.upsamples_4.forward(h);
        h = self.upsamples_5.forward(h);
        h = self.upsamples_6.forward(h);
        h = self.upsamples_7.forward_up(h);
        h = self.upsamples_8.forward(h);
        h = self.upsamples_9.forward(h);
        h = self.upsamples_10.forward(h);
        h = self.upsamples_11.forward(h);
        h = self.upsamples_12.forward(h);
        h = self.upsamples_13.forward(h);
        h = self.upsamples_14.forward(h);
        self.head.forward(h)
    }
}

/// Complete WAN 2.1 Video VAE
#[derive(Module, Debug)]
pub struct WanVae<B: Backend> {
    pub encoder: WanVaeEncoder<B>,
    pub decoder: WanVaeDecoder<B>,
    /// Quant conv 32->32
    conv1: Conv3d<B>,
    /// Quant conv 16->16
    conv2: Conv3d<B>,
}

impl<B: Backend> WanVae<B> {
    pub fn new(config: &WanVaeConfig, device: &B::Device) -> Self {
        Self {
            encoder: WanVaeEncoder::new(config, device),
            decoder: WanVaeDecoder::new(config, device),
            conv1: Conv3dConfig::new([32, 32], [1, 1, 1]).init(device),
            conv2: Conv3dConfig::new([16, 16], [1, 1, 1]).init(device),
        }
    }

    /// Encode video to latent space
    pub fn encode(&self, video: Tensor<B, 5>) -> Tensor<B, 5> {
        let x = video * 2.0 - 1.0;
        let h = self.encoder.forward(x);
        let h = self.conv1.forward(h);
        let [batch, _channels, time, height, width] = h.dims();
        h.slice([0..batch, 0..16, 0..time, 0..height, 0..width])
    }

    /// Decode latents to video
    pub fn decode(&self, latents: Tensor<B, 5>) -> Tensor<B, 5> {
        let z = self.conv2.forward(latents);
        let x = self.decoder.forward(z);
        (x + 1.0) / 2.0
    }

    /// Encode video to latent space (deterministic, no sampling)
    pub fn encode_deterministic(&self, video: Tensor<B, 5>) -> Tensor<B, 5> {
        self.encode(video)
    }

    /// Get the compression ratios
    pub fn compression_ratio(&self) -> (usize, usize, usize) {
        (4, 8, 8) // time, height, width
    }

    /// Get the latent channels
    pub fn latent_channels(&self) -> usize {
        16
    }
}

/// Create a SafetensorsStore with WAN VAE key remapping
fn create_vae_safetensors_store(path: PathBuf) -> SafetensorsStore {
    SafetensorsStore::from_file(path)
        .with_from_adapter(PyTorchToBurnAdapter::default())
        // Encoder conv1
        .with_key_remapping(r"^encoder\.conv1\.", "encoder.conv1.")
        // Encoder downsamples - residual blocks with indexed naming
        .with_key_remapping(r"^encoder\.downsamples\.0\.residual\.0\.gamma$", "encoder.downsamples_0.residual.norm1.gamma")
        .with_key_remapping(r"^encoder\.downsamples\.0\.residual\.2\.", "encoder.downsamples_0.residual.conv1.")
        .with_key_remapping(r"^encoder\.downsamples\.0\.residual\.3\.gamma$", "encoder.downsamples_0.residual.norm2.gamma")
        .with_key_remapping(r"^encoder\.downsamples\.0\.residual\.6\.", "encoder.downsamples_0.residual.conv2.")
        .with_key_remapping(r"^encoder\.downsamples\.1\.residual\.0\.gamma$", "encoder.downsamples_1.residual.norm1.gamma")
        .with_key_remapping(r"^encoder\.downsamples\.1\.residual\.2\.", "encoder.downsamples_1.residual.conv1.")
        .with_key_remapping(r"^encoder\.downsamples\.1\.residual\.3\.gamma$", "encoder.downsamples_1.residual.norm2.gamma")
        .with_key_remapping(r"^encoder\.downsamples\.1\.residual\.6\.", "encoder.downsamples_1.residual.conv2.")
        // Encoder downsamples.2 - spatial resample
        .with_key_remapping(r"^encoder\.downsamples\.2\.resample\.1\.", "encoder.downsamples_2.resample.conv.")
        // Encoder downsamples.3 - residual with shortcut
        .with_key_remapping(r"^encoder\.downsamples\.3\.residual\.0\.gamma$", "encoder.downsamples_3.residual.norm1.gamma")
        .with_key_remapping(r"^encoder\.downsamples\.3\.residual\.2\.", "encoder.downsamples_3.residual.conv1.")
        .with_key_remapping(r"^encoder\.downsamples\.3\.residual\.3\.gamma$", "encoder.downsamples_3.residual.norm2.gamma")
        .with_key_remapping(r"^encoder\.downsamples\.3\.residual\.6\.", "encoder.downsamples_3.residual.conv2.")
        .with_key_remapping(r"^encoder\.downsamples\.3\.shortcut\.", "encoder.downsamples_3.shortcut.")
        // Encoder downsamples.4
        .with_key_remapping(r"^encoder\.downsamples\.4\.residual\.0\.gamma$", "encoder.downsamples_4.residual.norm1.gamma")
        .with_key_remapping(r"^encoder\.downsamples\.4\.residual\.2\.", "encoder.downsamples_4.residual.conv1.")
        .with_key_remapping(r"^encoder\.downsamples\.4\.residual\.3\.gamma$", "encoder.downsamples_4.residual.norm2.gamma")
        .with_key_remapping(r"^encoder\.downsamples\.4\.residual\.6\.", "encoder.downsamples_4.residual.conv2.")
        // Encoder downsamples.5 - spatio-temporal resample
        .with_key_remapping(r"^encoder\.downsamples\.5\.resample\.1\.", "encoder.downsamples_5.resample.resample.conv.")
        .with_key_remapping(r"^encoder\.downsamples\.5\.time_conv\.", "encoder.downsamples_5.time_conv.time_conv.")
        // Encoder downsamples.6 with shortcut
        .with_key_remapping(r"^encoder\.downsamples\.6\.residual\.0\.gamma$", "encoder.downsamples_6.residual.norm1.gamma")
        .with_key_remapping(r"^encoder\.downsamples\.6\.residual\.2\.", "encoder.downsamples_6.residual.conv1.")
        .with_key_remapping(r"^encoder\.downsamples\.6\.residual\.3\.gamma$", "encoder.downsamples_6.residual.norm2.gamma")
        .with_key_remapping(r"^encoder\.downsamples\.6\.residual\.6\.", "encoder.downsamples_6.residual.conv2.")
        .with_key_remapping(r"^encoder\.downsamples\.6\.shortcut\.", "encoder.downsamples_6.shortcut.")
        // Encoder downsamples.7
        .with_key_remapping(r"^encoder\.downsamples\.7\.residual\.0\.gamma$", "encoder.downsamples_7.residual.norm1.gamma")
        .with_key_remapping(r"^encoder\.downsamples\.7\.residual\.2\.", "encoder.downsamples_7.residual.conv1.")
        .with_key_remapping(r"^encoder\.downsamples\.7\.residual\.3\.gamma$", "encoder.downsamples_7.residual.norm2.gamma")
        .with_key_remapping(r"^encoder\.downsamples\.7\.residual\.6\.", "encoder.downsamples_7.residual.conv2.")
        // Encoder downsamples.8 - spatio-temporal resample
        .with_key_remapping(r"^encoder\.downsamples\.8\.resample\.1\.", "encoder.downsamples_8.resample.resample.conv.")
        .with_key_remapping(r"^encoder\.downsamples\.8\.time_conv\.", "encoder.downsamples_8.time_conv.time_conv.")
        // Encoder downsamples.9
        .with_key_remapping(r"^encoder\.downsamples\.9\.residual\.0\.gamma$", "encoder.downsamples_9.residual.norm1.gamma")
        .with_key_remapping(r"^encoder\.downsamples\.9\.residual\.2\.", "encoder.downsamples_9.residual.conv1.")
        .with_key_remapping(r"^encoder\.downsamples\.9\.residual\.3\.gamma$", "encoder.downsamples_9.residual.norm2.gamma")
        .with_key_remapping(r"^encoder\.downsamples\.9\.residual\.6\.", "encoder.downsamples_9.residual.conv2.")
        // Encoder downsamples.10
        .with_key_remapping(r"^encoder\.downsamples\.10\.residual\.0\.gamma$", "encoder.downsamples_10.residual.norm1.gamma")
        .with_key_remapping(r"^encoder\.downsamples\.10\.residual\.2\.", "encoder.downsamples_10.residual.conv1.")
        .with_key_remapping(r"^encoder\.downsamples\.10\.residual\.3\.gamma$", "encoder.downsamples_10.residual.norm2.gamma")
        .with_key_remapping(r"^encoder\.downsamples\.10\.residual\.6\.", "encoder.downsamples_10.residual.conv2.")
        // Encoder middle blocks
        .with_key_remapping(r"^encoder\.middle\.0\.residual\.0\.gamma$", "encoder.middle_0.residual.norm1.gamma")
        .with_key_remapping(r"^encoder\.middle\.0\.residual\.2\.", "encoder.middle_0.residual.conv1.")
        .with_key_remapping(r"^encoder\.middle\.0\.residual\.3\.gamma$", "encoder.middle_0.residual.norm2.gamma")
        .with_key_remapping(r"^encoder\.middle\.0\.residual\.6\.", "encoder.middle_0.residual.conv2.")
        .with_key_remapping(r"^encoder\.middle\.1\.norm\.gamma$", "encoder.middle_1.norm.gamma")
        .with_key_remapping(r"^encoder\.middle\.1\.to_qkv\.", "encoder.middle_1.to_qkv.")
        .with_key_remapping(r"^encoder\.middle\.1\.proj\.", "encoder.middle_1.proj.")
        .with_key_remapping(r"^encoder\.middle\.2\.residual\.0\.gamma$", "encoder.middle_2.residual.norm1.gamma")
        .with_key_remapping(r"^encoder\.middle\.2\.residual\.2\.", "encoder.middle_2.residual.conv1.")
        .with_key_remapping(r"^encoder\.middle\.2\.residual\.3\.gamma$", "encoder.middle_2.residual.norm2.gamma")
        .with_key_remapping(r"^encoder\.middle\.2\.residual\.6\.", "encoder.middle_2.residual.conv2.")
        // Encoder head
        .with_key_remapping(r"^encoder\.head\.0\.gamma$", "encoder.head.head.norm.gamma")
        .with_key_remapping(r"^encoder\.head\.2\.", "encoder.head.head.conv.")
        // Decoder conv1
        .with_key_remapping(r"^decoder\.conv1\.", "decoder.conv1.")
        // Decoder middle blocks
        .with_key_remapping(r"^decoder\.middle\.0\.residual\.0\.gamma$", "decoder.middle_0.residual.norm1.gamma")
        .with_key_remapping(r"^decoder\.middle\.0\.residual\.2\.", "decoder.middle_0.residual.conv1.")
        .with_key_remapping(r"^decoder\.middle\.0\.residual\.3\.gamma$", "decoder.middle_0.residual.norm2.gamma")
        .with_key_remapping(r"^decoder\.middle\.0\.residual\.6\.", "decoder.middle_0.residual.conv2.")
        .with_key_remapping(r"^decoder\.middle\.1\.norm\.gamma$", "decoder.middle_1.norm.gamma")
        .with_key_remapping(r"^decoder\.middle\.1\.to_qkv\.", "decoder.middle_1.to_qkv.")
        .with_key_remapping(r"^decoder\.middle\.1\.proj\.", "decoder.middle_1.proj.")
        .with_key_remapping(r"^decoder\.middle\.2\.residual\.0\.gamma$", "decoder.middle_2.residual.norm1.gamma")
        .with_key_remapping(r"^decoder\.middle\.2\.residual\.2\.", "decoder.middle_2.residual.conv1.")
        .with_key_remapping(r"^decoder\.middle\.2\.residual\.3\.gamma$", "decoder.middle_2.residual.norm2.gamma")
        .with_key_remapping(r"^decoder\.middle\.2\.residual\.6\.", "decoder.middle_2.residual.conv2.")
        // Decoder upsamples.0-2 (residual blocks)
        .with_key_remapping(r"^decoder\.upsamples\.0\.residual\.0\.gamma$", "decoder.upsamples_0.residual.norm1.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.0\.residual\.2\.", "decoder.upsamples_0.residual.conv1.")
        .with_key_remapping(r"^decoder\.upsamples\.0\.residual\.3\.gamma$", "decoder.upsamples_0.residual.norm2.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.0\.residual\.6\.", "decoder.upsamples_0.residual.conv2.")
        .with_key_remapping(r"^decoder\.upsamples\.1\.residual\.0\.gamma$", "decoder.upsamples_1.residual.norm1.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.1\.residual\.2\.", "decoder.upsamples_1.residual.conv1.")
        .with_key_remapping(r"^decoder\.upsamples\.1\.residual\.3\.gamma$", "decoder.upsamples_1.residual.norm2.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.1\.residual\.6\.", "decoder.upsamples_1.residual.conv2.")
        .with_key_remapping(r"^decoder\.upsamples\.2\.residual\.0\.gamma$", "decoder.upsamples_2.residual.norm1.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.2\.residual\.2\.", "decoder.upsamples_2.residual.conv1.")
        .with_key_remapping(r"^decoder\.upsamples\.2\.residual\.3\.gamma$", "decoder.upsamples_2.residual.norm2.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.2\.residual\.6\.", "decoder.upsamples_2.residual.conv2.")
        // Decoder upsamples.3 - spatio-temporal resample
        .with_key_remapping(r"^decoder\.upsamples\.3\.resample\.1\.", "decoder.upsamples_3.resample.resample.conv.")
        .with_key_remapping(r"^decoder\.upsamples\.3\.time_conv\.", "decoder.upsamples_3.time_conv.time_conv.")
        // Decoder upsamples.4 with shortcut
        .with_key_remapping(r"^decoder\.upsamples\.4\.residual\.0\.gamma$", "decoder.upsamples_4.residual.norm1.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.4\.residual\.2\.", "decoder.upsamples_4.residual.conv1.")
        .with_key_remapping(r"^decoder\.upsamples\.4\.residual\.3\.gamma$", "decoder.upsamples_4.residual.norm2.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.4\.residual\.6\.", "decoder.upsamples_4.residual.conv2.")
        .with_key_remapping(r"^decoder\.upsamples\.4\.shortcut\.", "decoder.upsamples_4.shortcut.")
        // Decoder upsamples.5-6
        .with_key_remapping(r"^decoder\.upsamples\.5\.residual\.0\.gamma$", "decoder.upsamples_5.residual.norm1.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.5\.residual\.2\.", "decoder.upsamples_5.residual.conv1.")
        .with_key_remapping(r"^decoder\.upsamples\.5\.residual\.3\.gamma$", "decoder.upsamples_5.residual.norm2.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.5\.residual\.6\.", "decoder.upsamples_5.residual.conv2.")
        .with_key_remapping(r"^decoder\.upsamples\.6\.residual\.0\.gamma$", "decoder.upsamples_6.residual.norm1.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.6\.residual\.2\.", "decoder.upsamples_6.residual.conv1.")
        .with_key_remapping(r"^decoder\.upsamples\.6\.residual\.3\.gamma$", "decoder.upsamples_6.residual.norm2.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.6\.residual\.6\.", "decoder.upsamples_6.residual.conv2.")
        // Decoder upsamples.7 - spatio-temporal resample
        .with_key_remapping(r"^decoder\.upsamples\.7\.resample\.1\.", "decoder.upsamples_7.resample.resample.conv.")
        .with_key_remapping(r"^decoder\.upsamples\.7\.time_conv\.", "decoder.upsamples_7.time_conv.time_conv.")
        // Decoder upsamples.8-10
        .with_key_remapping(r"^decoder\.upsamples\.8\.residual\.0\.gamma$", "decoder.upsamples_8.residual.norm1.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.8\.residual\.2\.", "decoder.upsamples_8.residual.conv1.")
        .with_key_remapping(r"^decoder\.upsamples\.8\.residual\.3\.gamma$", "decoder.upsamples_8.residual.norm2.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.8\.residual\.6\.", "decoder.upsamples_8.residual.conv2.")
        .with_key_remapping(r"^decoder\.upsamples\.9\.residual\.0\.gamma$", "decoder.upsamples_9.residual.norm1.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.9\.residual\.2\.", "decoder.upsamples_9.residual.conv1.")
        .with_key_remapping(r"^decoder\.upsamples\.9\.residual\.3\.gamma$", "decoder.upsamples_9.residual.norm2.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.9\.residual\.6\.", "decoder.upsamples_9.residual.conv2.")
        .with_key_remapping(r"^decoder\.upsamples\.10\.residual\.0\.gamma$", "decoder.upsamples_10.residual.norm1.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.10\.residual\.2\.", "decoder.upsamples_10.residual.conv1.")
        .with_key_remapping(r"^decoder\.upsamples\.10\.residual\.3\.gamma$", "decoder.upsamples_10.residual.norm2.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.10\.residual\.6\.", "decoder.upsamples_10.residual.conv2.")
        // Decoder upsamples.11 - spatial resample
        .with_key_remapping(r"^decoder\.upsamples\.11\.resample\.1\.", "decoder.upsamples_11.resample.conv.")
        // Decoder upsamples.12-14
        .with_key_remapping(r"^decoder\.upsamples\.12\.residual\.0\.gamma$", "decoder.upsamples_12.residual.norm1.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.12\.residual\.2\.", "decoder.upsamples_12.residual.conv1.")
        .with_key_remapping(r"^decoder\.upsamples\.12\.residual\.3\.gamma$", "decoder.upsamples_12.residual.norm2.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.12\.residual\.6\.", "decoder.upsamples_12.residual.conv2.")
        .with_key_remapping(r"^decoder\.upsamples\.13\.residual\.0\.gamma$", "decoder.upsamples_13.residual.norm1.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.13\.residual\.2\.", "decoder.upsamples_13.residual.conv1.")
        .with_key_remapping(r"^decoder\.upsamples\.13\.residual\.3\.gamma$", "decoder.upsamples_13.residual.norm2.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.13\.residual\.6\.", "decoder.upsamples_13.residual.conv2.")
        .with_key_remapping(r"^decoder\.upsamples\.14\.residual\.0\.gamma$", "decoder.upsamples_14.residual.norm1.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.14\.residual\.2\.", "decoder.upsamples_14.residual.conv1.")
        .with_key_remapping(r"^decoder\.upsamples\.14\.residual\.3\.gamma$", "decoder.upsamples_14.residual.norm2.gamma")
        .with_key_remapping(r"^decoder\.upsamples\.14\.residual\.6\.", "decoder.upsamples_14.residual.conv2.")
        // Decoder head
        .with_key_remapping(r"^decoder\.head\.0\.gamma$", "decoder.head.head.norm.gamma")
        .with_key_remapping(r"^decoder\.head\.2\.", "decoder.head.head.conv.")
}

#[derive(Debug, thiserror::Error)]
pub enum VaeLoadError {
    #[error("Error while loading weights: {0}")]
    LoadError(String),
    #[error("Unrecognised file extension")]
    UnknownExtension,
}

impl<B: Backend> WanVae<B> {
    pub fn load_weights(&mut self, path: impl Into<PathBuf>) -> Result<(), VaeLoadError> {
        let path = path.into();
        let extension = path.extension().map(|s| s.to_string_lossy().to_lowercase());

        match extension.as_deref() {
            Some("safetensors") => {
                eprintln!("[wan-vae] Loading weights from safetensors...");
                let mut weights = create_vae_safetensors_store(path);
                weights
                    .apply_to(self)
                    .map_err(|e| VaeLoadError::LoadError(e.to_string()))?;
            }
            Some("bpk") | None => {
                eprintln!("[wan-vae] Loading weights from bpk...");
                let mut weights = BurnpackStore::from_file(path).auto_extension(false);
                weights
                    .apply_to(self)
                    .map_err(|e| VaeLoadError::LoadError(e.to_string()))?;
            }
            _ => {
                return Err(VaeLoadError::UnknownExtension);
            }
        }

        Ok(())
    }

    pub fn with_weights(mut self, path: impl Into<PathBuf>) -> Result<Self, VaeLoadError> {
        self.load_weights(path)?;
        Ok(self)
    }
}

impl WanVaeConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> WanVae<B> {
        WanVae::new(self, device)
    }
}
