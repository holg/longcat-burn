//! Flow Matching Scheduler for LongCat
//!
//! Implements the flow matching framework for diffusion sampling.
//! Uses velocity prediction: v_t = dx_t/dt where x_t interpolates between noise and data.

use burn::prelude::*;

/// Flow Matching Scheduler
///
/// Implements the rectified flow / flow matching framework used by LongCat.
/// The forward process interpolates linearly between data x_0 and noise x_1:
///   x_t = (1 - t) * x_0 + t * x_1
///
/// The model predicts velocity v_t = dx_t/dt = x_1 - x_0
#[derive(Debug, Clone)]
pub struct FlowMatchingScheduler {
    /// Number of inference steps
    pub num_steps: usize,
    /// Timesteps for sampling (from 1.0 to 0.0)
    pub timesteps: Vec<f32>,
    /// Timestep shift for adaptive noise scheduling
    pub shift: f32,
}

impl FlowMatchingScheduler {
    /// Create a new flow matching scheduler
    ///
    /// # Arguments
    /// * `num_steps` - Number of sampling steps
    pub fn new(num_steps: usize) -> Self {
        Self::with_shift(num_steps, 1.0)
    }

    /// Create a scheduler with adaptive timestep shift
    ///
    /// Higher shift values bias towards higher noise levels, which is
    /// beneficial for higher resolution / longer videos.
    ///
    /// # Arguments
    /// * `num_steps` - Number of sampling steps
    /// * `shift` - Timestep shift factor (1.0 = no shift, higher = more noise)
    pub fn with_shift(num_steps: usize, shift: f32) -> Self {
        // Generate timesteps from 1.0 to 0.0 with shift
        let timesteps: Vec<f32> = (0..=num_steps)
            .map(|i| {
                let t = 1.0 - (i as f32) / (num_steps as f32);
                // Apply shift: t' = t^shift
                t.powf(shift)
            })
            .collect();

        Self {
            num_steps,
            timesteps,
            shift,
        }
    }

    /// Create scheduler with adaptive shift based on video dimensions
    ///
    /// LongCat adaptively adjusts shift based on the number of noise tokens,
    /// preferring higher noise levels for larger videos.
    ///
    /// # Arguments
    /// * `num_steps` - Number of sampling steps
    /// * `num_tokens` - Number of latent tokens (t_patches * h_patches * w_patches)
    pub fn with_adaptive_shift(num_steps: usize, num_tokens: usize) -> Self {
        // Heuristic: increase shift for larger videos
        // Base shift is 1.0, increases logarithmically with token count
        let base_tokens = 1000; // Approximate tokens for 480p short video
        let shift = 1.0 + 0.5 * (num_tokens as f32 / base_tokens as f32).ln().max(0.0);
        Self::with_shift(num_steps, shift.min(3.0))
    }

    /// Get the timesteps for sampling
    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    /// Get timestep at a given step index
    pub fn get_timestep(&self, step: usize) -> f32 {
        self.timesteps[step]
    }

    /// Sample initial noise
    pub fn sample_noise<B: Backend>(&self, shape: [usize; 5], device: &B::Device) -> Tensor<B, 5> {
        Tensor::random(shape, burn::tensor::Distribution::Normal(0.0, 1.0), device)
    }

    /// Sample initial noise with seed
    pub fn sample_noise_seeded<B: Backend>(
        &self,
        shape: [usize; 5],
        _seed: u64,
        device: &B::Device,
    ) -> Tensor<B, 5> {
        // Note: Burn doesn't support seeded random tensors directly
        // For reproducibility, you'd need to use a custom RNG
        Tensor::random(shape, burn::tensor::Distribution::Normal(0.0, 1.0), device)
    }

    /// Perform one sampling step
    ///
    /// Given x_t and predicted velocity v_t, compute x_{t-dt}
    ///
    /// # Arguments
    /// * `x_t` - Current noisy sample at timestep t
    /// * `velocity` - Predicted velocity v_t from the model
    /// * `step` - Current step index (0 = most noisy)
    ///
    /// # Returns
    /// Sample at the next timestep (less noisy)
    pub fn step<B: Backend>(
        &self,
        x_t: Tensor<B, 5>,
        velocity: Tensor<B, 5>,
        step: usize,
    ) -> Tensor<B, 5> {
        let t_current = self.timesteps[step];
        let t_next = self.timesteps[step + 1];
        let dt = t_next - t_current; // Negative, since we're going from 1 to 0

        // Euler step: x_{t+dt} = x_t + dt * v_t
        x_t + velocity * dt
    }

    /// Perform one sampling step with classifier-free guidance
    ///
    /// # Arguments
    /// * `x_t` - Current noisy sample
    /// * `velocity_cond` - Velocity prediction with conditioning
    /// * `velocity_uncond` - Velocity prediction without conditioning
    /// * `guidance_scale` - CFG scale (1.0 = no guidance)
    /// * `step` - Current step index
    pub fn step_cfg<B: Backend>(
        &self,
        x_t: Tensor<B, 5>,
        velocity_cond: Tensor<B, 5>,
        velocity_uncond: Tensor<B, 5>,
        guidance_scale: f32,
        step: usize,
    ) -> Tensor<B, 5> {
        // Apply classifier-free guidance
        let velocity = velocity_uncond.clone()
            + (velocity_cond - velocity_uncond) * guidance_scale;

        self.step(x_t, velocity, step)
    }

    /// Add noise to samples (for training or testing)
    ///
    /// # Arguments
    /// * `x_0` - Clean samples
    /// * `noise` - Random noise (same shape as x_0)
    /// * `t` - Timestep in [0, 1]
    pub fn add_noise<B: Backend>(
        &self,
        x_0: Tensor<B, 5>,
        noise: Tensor<B, 5>,
        t: f32,
    ) -> Tensor<B, 5> {
        // Linear interpolation: x_t = (1 - t) * x_0 + t * noise
        x_0 * (1.0 - t) + noise * t
    }

    /// Get velocity target for training
    ///
    /// For flow matching, the velocity target is simply: v = noise - x_0
    ///
    /// # Arguments
    /// * `x_0` - Clean samples
    /// * `noise` - Random noise
    pub fn get_velocity_target<B: Backend>(
        &self,
        x_0: Tensor<B, 5>,
        noise: Tensor<B, 5>,
    ) -> Tensor<B, 5> {
        noise - x_0
    }

    /// Sample random timesteps for training
    ///
    /// Uses logit-normal-like weighting for stable training.
    ///
    /// # Arguments
    /// * `batch_size` - Number of timesteps to sample
    /// * `device` - Device to create tensor on
    pub fn sample_timesteps<B: Backend>(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        // Sample from uniform distribution [0, 1]
        Tensor::random([batch_size], burn::tensor::Distribution::Uniform(0.0, 1.0), device)
    }

    /// Sample timesteps with logit-normal-like weighting
    ///
    /// This biases sampling towards the middle of the [0, 1] range,
    /// which the LongCat paper found to be more stable.
    ///
    /// # Arguments
    /// * `batch_size` - Number of timesteps to sample
    /// * `mean` - Mean of the logit-normal (default: 0.0)
    /// * `std` - Std of the logit-normal (default: 1.0)
    /// * `device` - Device to create tensor on
    pub fn sample_timesteps_logit_normal<B: Backend>(
        &self,
        batch_size: usize,
        mean: f32,
        std: f32,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        // Sample from normal distribution
        let z = Tensor::<B, 1>::random(
            [batch_size],
            burn::tensor::Distribution::Normal(mean.into(), std.into()),
            device,
        );

        // Apply sigmoid to get [0, 1]
        burn::tensor::activation::sigmoid(z)
    }
}

/// Configuration for the scheduler
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Number of inference steps
    pub num_steps: usize,
    /// Timestep shift (1.0 = no shift)
    pub shift: f32,
    /// Whether to use adaptive shift based on video size
    pub adaptive_shift: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_steps: 50,
            shift: 1.0,
            adaptive_shift: true,
        }
    }
}

impl SchedulerConfig {
    /// Create scheduler with default settings
    pub fn new(num_steps: usize) -> Self {
        Self {
            num_steps,
            ..Default::default()
        }
    }

    /// Build the scheduler
    pub fn build(&self) -> FlowMatchingScheduler {
        FlowMatchingScheduler::with_shift(self.num_steps, self.shift)
    }

    /// Build scheduler with adaptive shift for given video dimensions
    pub fn build_adaptive(&self, num_tokens: usize) -> FlowMatchingScheduler {
        if self.adaptive_shift {
            FlowMatchingScheduler::with_adaptive_shift(self.num_steps, num_tokens)
        } else {
            FlowMatchingScheduler::with_shift(self.num_steps, self.shift)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_creation() {
        let scheduler = FlowMatchingScheduler::new(50);
        assert_eq!(scheduler.num_steps, 50);
        assert_eq!(scheduler.timesteps.len(), 51);
        assert!((scheduler.timesteps[0] - 1.0).abs() < 1e-6);
        assert!((scheduler.timesteps[50] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_scheduler_with_shift() {
        let scheduler = FlowMatchingScheduler::with_shift(10, 2.0);
        // With shift=2.0, timesteps should be more concentrated near 0
        assert!(scheduler.timesteps[5] < 0.5); // Midpoint should be less than 0.5
    }

    #[test]
    fn test_adaptive_shift() {
        let small = FlowMatchingScheduler::with_adaptive_shift(50, 500);
        let large = FlowMatchingScheduler::with_adaptive_shift(50, 5000);
        // Larger videos should have higher shift
        assert!(large.shift > small.shift);
    }
}
