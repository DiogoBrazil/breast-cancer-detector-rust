use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn::data::dataset::Dataset;

use super::annotation::{AnnotationFile, BoundingBox};

#[derive(Debug, Clone)]
pub struct MammogramSample {
    pub image_path: PathBuf,
    pub image_name: String,
    pub width: u32,
    pub height: u32,
    pub boxes: Vec<BoundingBox>,
}

#[derive(Debug, Clone)]
pub struct MammogramDataset {
    samples: Vec<MammogramSample>,
}

impl MammogramDataset {
    pub fn new(split_dir: impl AsRef<Path>) -> Result<Self> {
        let split_dir = split_dir.as_ref();
        let labels_dir = split_dir.join("labels_json");

        if !split_dir.exists() {
            anyhow::bail!("Split directory not found: {}", split_dir.display());
        }

        if !labels_dir.exists() {
            anyhow::bail!("Labels directory not found: {}", labels_dir.display());
        }

        let mut samples = Vec::new();

        for entry in fs::read_dir(&labels_dir)
            .with_context(|| format!("Failed to read labels dir {}", labels_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }

            let content = fs::read_to_string(&path)
                .with_context(|| format!("Failed to read annotation file {}", path.display()))?;

            let ann: AnnotationFile = serde_json::from_str(&content)
                .with_context(|| format!("Failed to parse annotation file {}", path.display()))?;

            let image_path = split_dir.join(&ann.image);

            if !image_path.exists() {
                anyhow::bail!("Image file not found: {}", image_path.display());
            }

            let valid_boxes = ann
                .boxes
                .into_iter()
                .filter(|b| b.is_valid())
                .collect::<Vec<BoundingBox>>();

            samples.push(MammogramSample {
                image_path,
                image_name: ann.image,
                width: ann.width,
                height: ann.height,
                boxes: valid_boxes,
            });
        }

        samples.sort_by(|a, b| a.image_name.cmp(&b.image_name));

        Ok(Self { samples })
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

impl Dataset<MammogramSample> for MammogramDataset {
    fn get(&self, index: usize) -> Option<MammogramSample> {
        self.samples.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}
