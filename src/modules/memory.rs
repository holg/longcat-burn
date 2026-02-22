//! Memory optimization utilities for LongCat
//!
//! Provides global configuration for memory-efficient inference.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Global attention slice size (0 = no slicing, compute full attention)
/// Smaller values use less memory but are slower.
/// Recommended values:
/// - 0: Full attention (fastest, most memory)
/// - 2048: Good balance for 32GB+ VRAM
/// - 1024: For 16-24GB VRAM
/// - 512: For 8-16GB VRAM
/// - 256: For <8GB VRAM (very slow)
static ATTENTION_SLICE_SIZE: AtomicUsize = AtomicUsize::new(0);

/// Whether to use sequential block processing (saves memory, slower)
static SEQUENTIAL_BLOCKS: AtomicUsize = AtomicUsize::new(0);

/// Set the attention slice size for memory optimization.
///
/// When set to a value > 0, attention is computed in chunks of this size
/// rather than all at once, significantly reducing peak memory usage.
///
/// # Arguments
/// * `size` - Slice size (0 = full attention, >0 = chunked)
pub fn set_attention_slice_size(size: usize) {
    ATTENTION_SLICE_SIZE.store(size, Ordering::Relaxed);
    if size > 0 {
        eprintln!("[longcat] Attention slice size set to {} (memory optimization enabled)", size);
    } else {
        eprintln!("[longcat] Attention slicing disabled (full attention)");
    }
}

/// Get the current attention slice size.
pub fn get_attention_slice_size() -> usize {
    ATTENTION_SLICE_SIZE.load(Ordering::Relaxed)
}

/// Enable or disable sequential block processing.
///
/// When enabled, transformer blocks are processed one at a time with
/// intermediate tensors freed immediately, reducing peak memory.
pub fn set_sequential_blocks(enabled: bool) {
    SEQUENTIAL_BLOCKS.store(if enabled { 1 } else { 0 }, Ordering::Relaxed);
}

/// Check if sequential block processing is enabled.
pub fn get_sequential_blocks() -> bool {
    SEQUENTIAL_BLOCKS.load(Ordering::Relaxed) != 0
}

/// Memory configuration for video generation
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Attention slice size (0 = full attention)
    pub attention_slice_size: usize,
    /// Use sequential block processing
    pub sequential_blocks: bool,
    /// Enable VAE tiling for large videos
    pub vae_tiling: bool,
    /// VAE tile size (frames per tile)
    pub vae_tile_frames: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            attention_slice_size: 0,
            sequential_blocks: false,
            vae_tiling: false,
            vae_tile_frames: 16,
        }
    }
}

impl MemoryConfig {
    /// Low memory configuration for systems with <16GB VRAM
    pub fn low_memory() -> Self {
        Self {
            attention_slice_size: 512,
            sequential_blocks: true,
            vae_tiling: true,
            vae_tile_frames: 8,
        }
    }

    /// Medium memory configuration for 16-32GB VRAM
    pub fn medium_memory() -> Self {
        Self {
            attention_slice_size: 1024,
            sequential_blocks: false,
            vae_tiling: true,
            vae_tile_frames: 16,
        }
    }

    /// High memory configuration for 32GB+ VRAM
    pub fn high_memory() -> Self {
        Self {
            attention_slice_size: 2048,
            sequential_blocks: false,
            vae_tiling: false,
            vae_tile_frames: 32,
        }
    }

    /// Apply this configuration globally
    pub fn apply(&self) {
        set_attention_slice_size(self.attention_slice_size);
        set_sequential_blocks(self.sequential_blocks);
    }
}
