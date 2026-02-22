//! Weight loading utilities for LongCat models.

use std::path::PathBuf;

use burn::{
    prelude::Backend,
    store::{BurnpackStore, ModuleStore, PyTorchToBurnAdapter, SafetensorsStore},
};
use thiserror::Error;

use crate::modules::dit::LongCatDiT;

#[derive(Error, Debug)]
pub enum ModelLoadError {
    #[error("Error while loading weights: {0}")]
    LoadError(String),
    #[error("Unrecognised file extension")]
    UnknownExtension,
}

/// Create a SafetensorsStore with ComfyUI/Kijai key remapping
///
/// Maps from safetensors keys to Burn module structure:
/// - patch_embedding.weight/bias -> patch_embedding.weight/bias
/// - time_embedding.mlp.0/2 -> time_embedding.mlp.linear1/linear2
/// - text_embedding.0/2 -> text_embedding.linear1/linear2
/// - blocks.X.self_attn.q/k/v/o -> blocks.X.self_attn.q/k/v/o
/// - blocks.X.self_attn.norm_q/norm_k -> blocks.X.self_attn.norm_q/norm_k
/// - blocks.X.cross_attn.q/k/v/o -> blocks.X.cross_attn.q/k/v/o
/// - blocks.X.cross_attn.norm_q/norm_k -> blocks.X.cross_attn.norm_q/norm_k
/// - blocks.X.ffn.w1/w2/w3 -> blocks.X.ffn.w1/w2/w3
/// - blocks.X.norm3 -> blocks.X.norm3
/// - blocks.X.modulation.1 -> blocks.X.modulation
/// - head.head -> head.head
/// - head.modulation.1 -> head.modulation
fn create_safetensors_store(path: PathBuf) -> SafetensorsStore {
    SafetensorsStore::from_file(path)
        .with_from_adapter(PyTorchToBurnAdapter::default())
        // Time embedding MLP
        .with_key_remapping(r"^time_embedding\.mlp\.0\.", "time_embedding.mlp.linear1.")
        .with_key_remapping(r"^time_embedding\.mlp\.2\.", "time_embedding.mlp.linear2.")
        // Text embedding MLP
        .with_key_remapping(r"^text_embedding\.0\.", "text_embedding.linear1.")
        .with_key_remapping(r"^text_embedding\.2\.", "text_embedding.linear2.")
        // Block modulation: modulation.1 -> modulation (Sequential index stripped)
        .with_key_remapping(r"^blocks\.(\d+)\.modulation\.1\.", "blocks.$1.modulation.")
        // Head modulation: head.modulation.1 -> head.modulation
        .with_key_remapping(r"^head\.modulation\.1\.", "head.modulation.")
        // Self-attention norm: norm_q/norm_k -> norm_q/norm_k (RmsNorm uses .gamma)
        .with_key_remapping(r"\.self_attn\.norm_q\.weight$", ".self_attn.norm_q.gamma")
        .with_key_remapping(r"\.self_attn\.norm_k\.weight$", ".self_attn.norm_k.gamma")
        // Cross-attention norm
        .with_key_remapping(r"\.cross_attn\.norm_q\.weight$", ".cross_attn.norm_q.gamma")
        .with_key_remapping(r"\.cross_attn\.norm_k\.weight$", ".cross_attn.norm_k.gamma")
        // Block norm3: weight -> gamma
        .with_key_remapping(r"\.norm3\.weight$", ".norm3.gamma")
        // Self-attention output: o -> o
        .with_key_remapping(r"\.self_attn\.o\.", ".self_attn.o.")
        // Cross-attention output: o -> o
        .with_key_remapping(r"\.cross_attn\.o\.", ".cross_attn.o.")
}

impl<B: Backend> LongCatDiT<B> {
    /// Load weights and return self (builder pattern)
    pub fn with_weights(
        mut self,
        path: impl Into<PathBuf>,
    ) -> Result<Self, ModelLoadError> {
        self.load_weights(path)?;
        Ok(self)
    }

    /// Load weights from a file
    ///
    /// Supports:
    /// - `.safetensors` - ComfyUI/Kijai format
    /// - `.bpk` - Burn native format
    pub fn load_weights(&mut self, path: impl Into<PathBuf>) -> Result<(), ModelLoadError> {
        let path = path.into();
        let extension = path.extension().map(|s| s.to_string_lossy().to_lowercase());

        match extension.as_deref() {
            Some("safetensors") => {
                eprintln!("[longcat] Loading weights from safetensors...");
                let mut weights = create_safetensors_store(path);
                weights.apply_to(self)
                    .map_err(|e| ModelLoadError::LoadError(e.to_string()))?;
            }
            Some("bpk") | None => {
                eprintln!("[longcat] Loading weights from bpk...");
                let mut weights = BurnpackStore::from_file(path)
                    .auto_extension(false);
                weights.apply_to(self)
                    .map_err(|e| ModelLoadError::LoadError(e.to_string()))?;
            }
            _ => {
                return Err(ModelLoadError::UnknownExtension);
            }
        }

        Ok(())
    }
}
