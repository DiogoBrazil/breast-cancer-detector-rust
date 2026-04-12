use anyhow::Result;
use burn::tensor::{Tensor, backend::Backend};
use image::GrayImage;

use super::annotation::BoundingBox;
use super::dataset::MammogramSample;
use super::image_ops::{augment, load_grayscale_image, preprocess_image_and_boxes};
use crate::training::config::AugmentConfig;

/// Amostra carregada do disco sem preprocessing (para augmentation on-the-fly).
#[derive(Clone)]
pub struct RawSample {
    pub gray_image: GrayImage,
    pub boxes: Vec<BoundingBox>,
}

/// Carrega imagem e boxes do disco sem aplicar resize/padding.
pub fn load_raw_sample(sample: &MammogramSample) -> Result<RawSample> {
    let gray_image = load_grayscale_image(&sample.image_path)?;
    Ok(RawSample {
        gray_image,
        boxes: sample.boxes.clone(),
    })
}

#[derive(Debug, Clone)]
pub struct PreparedSample {
    pub image_chw: Vec<f32>,
    pub boxes: Vec<[f32; 4]>,
    pub labels: Vec<usize>,
    pub image_size: usize,
}

/// Prepara um RawSample aplicando resize/padding para o tamanho alvo (sem augmentation).
pub fn prepare_from_raw(raw: &RawSample, image_size: usize) -> Result<PreparedSample> {
    let processed = preprocess_image_and_boxes(&raw.gray_image, &raw.boxes, image_size)?;
    Ok(PreparedSample {
        image_chw: processed.image_chw,
        boxes: processed.boxes,
        labels: processed.labels,
        image_size: processed.image_size,
    })
}

/// Aplica augmentation e depois preprocess.
pub fn prepare_from_raw_augmented(
    raw: &RawSample,
    image_size: usize,
    rng: &mut impl rand::Rng,
    augment_config: &AugmentConfig,
) -> Result<PreparedSample> {
    let (aug_image, aug_boxes) = augment(&raw.gray_image, &raw.boxes, rng, augment_config);
    let processed = preprocess_image_and_boxes(&aug_image, &aug_boxes, image_size)?;
    Ok(PreparedSample {
        image_chw: processed.image_chw,
        boxes: processed.boxes,
        labels: processed.labels,
        image_size: processed.image_size,
    })
}

pub fn prepare_sample(sample: &MammogramSample, image_size: usize) -> Result<PreparedSample> {
    let image = load_grayscale_image(&sample.image_path)?;
    let processed = preprocess_image_and_boxes(&image, &sample.boxes, image_size)?;

    Ok(PreparedSample {
        image_chw: processed.image_chw,
        boxes: processed.boxes,
        labels: processed.labels,
        image_size: processed.image_size,
    })
}

pub struct DetectionBatch<B: Backend> {
    pub images: Tensor<B, 4>,                    // [N, 3, H, W]
    pub boxes: Tensor<B, 3>,                     // [N, max_boxes, 4]
    pub labels: Tensor<B, 2, burn::tensor::Int>, // [N, max_boxes]
    pub mask: Tensor<B, 2>,                      // [N, max_boxes]
}

pub fn collate_detection_batch<B: Backend>(
    samples: &[PreparedSample],
    max_boxes: usize,
    device: &B::Device,
) -> DetectionBatch<B> {
    let batch_size = samples.len();
    let image_size = samples[0].image_size;

    let mut images = vec![0.0f32; batch_size * 3 * image_size * image_size];
    let mut boxes = vec![0.0f32; batch_size * max_boxes * 4];
    let mut labels = vec![0i64; batch_size * max_boxes];
    let mut mask = vec![0.0f32; batch_size * max_boxes];

    for (i, sample) in samples.iter().enumerate() {
        let img_offset = i * 3 * image_size * image_size;
        images[img_offset..img_offset + sample.image_chw.len()].copy_from_slice(&sample.image_chw);

        let valid_boxes = sample.boxes.len().min(max_boxes);

        for j in 0..valid_boxes {
            let b = sample.boxes[j];
            let box_offset = (i * max_boxes + j) * 4;
            boxes[box_offset] = b[0];
            boxes[box_offset + 1] = b[1];
            boxes[box_offset + 2] = b[2];
            boxes[box_offset + 3] = b[3];

            labels[i * max_boxes + j] = sample.labels[j] as i64;
            mask[i * max_boxes + j] = 1.0;
        }
    }

    let images = Tensor::<B, 1>::from_floats(images.as_slice(), device)
        .reshape([batch_size, 3, image_size, image_size]);

    let boxes =
        Tensor::<B, 1>::from_floats(boxes.as_slice(), device).reshape([batch_size, max_boxes, 4]);

    let labels = Tensor::<B, 1, burn::tensor::Int>::from_ints(labels.as_slice(), device)
        .reshape([batch_size, max_boxes]);

    let mask =
        Tensor::<B, 1>::from_floats(mask.as_slice(), device).reshape([batch_size, max_boxes]);

    DetectionBatch {
        images,
        boxes,
        labels,
        mask,
    }
}
