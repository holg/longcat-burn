//! 3D Rotary Position Embedding (RoPE) for LongCat
//!
//! Extends standard 1D RoPE to handle spatial-temporal dimensions (time, height, width).
//! Essential for understanding object movement across space and time.

use burn::prelude::*;

/// 3D Rotary Position Embedding
///
/// Applies rotary embeddings to queries and keys for spatial-temporal position encoding.
/// Features are paired into groups of 2, each group corresponds to one of the 3 dimensions.
///
/// Note: This is not a Module as it has no learnable parameters.
#[derive(Debug, Clone)]
pub struct Rope3D<B: Backend> {
    /// Base theta value for frequency computation
    theta: f64,
    /// Dimension allocation for each axis [time, height, width]
    dim_per_axis: [usize; 3],
    /// Phantom for backend type
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> Rope3D<B> {
    /// Create a new 3D RoPE module
    ///
    /// # Arguments
    /// * `head_dim` - Attention head dimension (e.g., 128)
    /// * `theta` - Base frequency (default: 10000.0)
    pub fn new(head_dim: usize, theta: f64) -> Self {
        // Distribute dimensions across 3 axes
        // Each axis dimension must be EVEN for RoPE (we split into pairs)
        // For head_dim=64: we want [22, 22, 20] or similar that sums to 64 with all even
        // For head_dim=128: [44, 42, 42] sums to 128

        // First, compute base dimension per axis (must be even)
        let base_dim = (head_dim / 3) & !1; // Round down to even
        let total_base = base_dim * 3;
        let remaining = head_dim - total_base;

        // Distribute remaining dimensions (add 2 at a time to keep even)
        let extra_pairs = remaining / 2;

        Self {
            theta,
            dim_per_axis: [
                base_dim + if extra_pairs > 0 { 2 } else { 0 },
                base_dim + if extra_pairs > 1 { 2 } else { 0 },
                base_dim + if extra_pairs > 2 { 2 } else { 0 },
            ],
            _backend: std::marker::PhantomData,
        }
    }

    /// Compute frequency table for a given dimension count
    fn compute_freqs(&self, dim: usize, device: &B::Device) -> Tensor<B, 1> {
        let half_dim = dim / 2;
        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / (self.theta as f32).powf(2.0 * i as f32 / dim as f32))
            .collect();
        Tensor::from_floats(freqs.as_slice(), device)
    }

    /// Compute rotary embeddings for positions along one axis
    fn compute_axis_rope(
        &self,
        positions: Tensor<B, 1, Int>,
        dim: usize,
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let freqs = self.compute_freqs(dim, device);
        let positions_float: Tensor<B, 1> = positions.float();

        // [seq_len, dim/2]
        let angles = positions_float.unsqueeze_dim(1) * freqs.unsqueeze_dim(0);

        (angles.clone().cos(), angles.sin())
    }

    /// Apply rotary embedding to a tensor
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, num_heads, seq_len, head_dim]
    /// * `t_pos` - Temporal positions [seq_len]
    /// * `h_pos` - Height positions [seq_len]
    /// * `w_pos` - Width positions [seq_len]
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        t_pos: Tensor<B, 1, Int>,
        h_pos: Tensor<B, 1, Int>,
        w_pos: Tensor<B, 1, Int>,
    ) -> Tensor<B, 4> {
        let device = x.device();
        let [batch, num_heads, _seq_len, _head_dim] = x.dims();

        // Compute RoPE for each axis
        let (cos_t, sin_t) = self.compute_axis_rope(t_pos, self.dim_per_axis[0], &device);
        let (cos_h, sin_h) = self.compute_axis_rope(h_pos, self.dim_per_axis[1], &device);
        let (cos_w, sin_w) = self.compute_axis_rope(w_pos, self.dim_per_axis[2], &device);

        // Concatenate cos/sin for all axes
        let cos = Tensor::cat(vec![cos_t, cos_h, cos_w], 1); // [seq_len, head_dim/2]
        let sin = Tensor::cat(vec![sin_t, sin_h, sin_w], 1);

        // Expand for batch and heads
        let cos = cos.unsqueeze_dims(&[0, 1]).repeat(&[batch, num_heads, 1, 1]);
        let sin = sin.unsqueeze_dims(&[0, 1]).repeat(&[batch, num_heads, 1, 1]);

        // Apply rotary embedding
        self.apply_rotary(x, cos, sin)
    }

    /// Apply rotary transformation
    /// x_rot = x * cos + rotate_half(x) * sin
    fn apply_rotary(
        &self,
        x: Tensor<B, 4>,
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let [batch, heads, seq, dim] = x.dims();
        let half_dim = dim / 2;

        // Split into two halves
        let x1 = x.clone().slice([0..batch, 0..heads, 0..seq, 0..half_dim]);
        let x2 = x.slice([0..batch, 0..heads, 0..seq, half_dim..dim]);

        // Rotate: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        let x1_rot = x1.clone() * cos.clone() - x2.clone() * sin.clone();
        let x2_rot = x1 * sin + x2 * cos;

        Tensor::cat(vec![x1_rot, x2_rot], 3)
    }
}

/// Precomputed 3D RoPE frequencies for a specific video shape
#[derive(Debug, Clone)]
pub struct Rope3DFreqs<B: Backend> {
    /// Cosine frequencies [seq_len, head_dim/2]
    pub cos: Tensor<B, 2>,
    /// Sine frequencies [seq_len, head_dim/2]
    pub sin: Tensor<B, 2>,
}

impl<B: Backend> Rope3DFreqs<B> {
    /// Precompute RoPE frequencies for a video of given dimensions
    ///
    /// # Arguments
    /// * `num_frames` - Number of temporal frames
    /// * `height` - Latent height (after VAE compression)
    /// * `width` - Latent width (after VAE compression)
    /// * `head_dim` - Attention head dimension
    /// * `theta` - Base frequency
    pub fn precompute(
        num_frames: usize,
        height: usize,
        width: usize,
        head_dim: usize,
        theta: f64,
        device: &B::Device,
    ) -> Self {
        let seq_len = num_frames * height * width;
        let rope = Rope3D::<B>::new(head_dim, theta);

        // Generate position indices for each dimension
        let mut t_positions = Vec::with_capacity(seq_len);
        let mut h_positions = Vec::with_capacity(seq_len);
        let mut w_positions = Vec::with_capacity(seq_len);

        for t in 0..num_frames {
            for h in 0..height {
                for w in 0..width {
                    t_positions.push(t as i64);
                    h_positions.push(h as i64);
                    w_positions.push(w as i64);
                }
            }
        }

        let t_pos = Tensor::<B, 1, Int>::from_ints(t_positions.as_slice(), device);
        let h_pos = Tensor::<B, 1, Int>::from_ints(h_positions.as_slice(), device);
        let w_pos = Tensor::<B, 1, Int>::from_ints(w_positions.as_slice(), device);

        // Compute frequencies for each axis
        let (cos_t, sin_t) = rope.compute_axis_rope(t_pos, rope.dim_per_axis[0], device);
        let (cos_h, sin_h) = rope.compute_axis_rope(h_pos, rope.dim_per_axis[1], device);
        let (cos_w, sin_w) = rope.compute_axis_rope(w_pos, rope.dim_per_axis[2], device);

        // Concatenate
        let cos = Tensor::cat(vec![cos_t, cos_h, cos_w], 1);
        let sin = Tensor::cat(vec![sin_t, sin_h, sin_w], 1);

        Self { cos, sin }
    }
}
