use anyhow::Result;
use burn::backend::Autodiff;
use burn::data::dataset::Dataset;
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::ElementConversion;
use burn::tensor::Tensor;
use burn_wgpu::{Wgpu, WgpuDevice};
use rand::seq::SliceRandom;
use rand::thread_rng;

use super::config::TrainConfig;
use super::loss::detection_loss;
use super::target::encode_targets;
use crate::data::batcher::{PreparedSample, prepare_sample};
use crate::data::dataset::MammogramDataset;
use crate::model::SimpleDetector;

/// Monta tensor de imagens [N, 3, H, W] a partir dos samples preparados.
fn build_image_tensor<B: burn::tensor::backend::Backend>(
    prepared: &[PreparedSample],
    image_size: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let batch_size = prepared.len();
    let pixels = 3 * image_size * image_size;
    let mut data = vec![0.0f32; batch_size * pixels];
    for (i, sample) in prepared.iter().enumerate() {
        let offset = i * pixels;
        data[offset..offset + sample.image_chw.len()].copy_from_slice(&sample.image_chw);
    }
    Tensor::<B, 1>::from_floats(data.as_slice(), device)
        .reshape([batch_size, 3, image_size, image_size])
}

pub fn run_train_loop(config: &TrainConfig) -> Result<()> {
    type InnerB = Wgpu;
    type B = Autodiff<InnerB>;

    let device = WgpuDevice::DefaultDevice;
    let grid_size = config.image_size / 32;

    // Dataset
    let train_dataset = MammogramDataset::new(format!("{}/train", config.dataset_root))?;
    let valid_dataset = MammogramDataset::new(format!("{}/valid", config.dataset_root))?;
    println!(
        "Train: {} samples | Valid: {} samples",
        train_dataset.len(),
        valid_dataset.len()
    );

    // Modelo e optimizer
    let mut model = SimpleDetector::<B>::new(&device, config.num_classes);
    let mut optim = AdamConfig::new().init();

    let mut rng = thread_rng();

    for epoch in 0..config.epochs {
        // === TRAIN ===
        let mut indices: Vec<usize> = (0..train_dataset.len()).collect();
        indices.shuffle(&mut rng);

        let mut train_cls = 0.0f64;
        let mut train_reg = 0.0f64;
        let mut train_batches = 0usize;

        for batch_start in (0..indices.len()).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(indices.len());
            let batch_indices = &indices[batch_start..batch_end];

            // Preparar samples (I/O + preprocess)
            let mut prepared = Vec::with_capacity(batch_indices.len());
            for &idx in batch_indices {
                let sample = train_dataset.get(idx).expect("sample should exist");
                prepared.push(prepare_sample(&sample, config.image_size)?);
            }

            let images = build_image_tensor::<B>(&prepared, config.image_size, &device);
            let targets = encode_targets::<B>(&prepared, config.image_size, grid_size, &device);

            // Forward + loss
            let output = model.forward(images);
            let losses = detection_loss(&output, &targets, 1.0);

            // Extrair valores antes do backward
            let cls_val: f32 = losses.cls_loss.clone().into_scalar().elem();
            let reg_val: f32 = losses.reg_loss.clone().into_scalar().elem();
            train_cls += cls_val as f64;
            train_reg += reg_val as f64;
            train_batches += 1;

            // Backward + optimizer step
            let grads = losses.total.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.learning_rate, model, grads);
        }

        let avg_train_cls = train_cls / train_batches as f64;
        let avg_train_reg = train_reg / train_batches as f64;

        // === VALIDATION ===
        let valid_model = model.valid();
        let mut valid_cls = 0.0f64;
        let mut valid_reg = 0.0f64;
        let mut valid_batches = 0usize;

        for batch_start in (0..valid_dataset.len()).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(valid_dataset.len());

            let mut prepared = Vec::with_capacity(batch_end - batch_start);
            for idx in batch_start..batch_end {
                let sample = valid_dataset.get(idx).expect("sample should exist");
                prepared.push(prepare_sample(&sample, config.image_size)?);
            }

            let images = build_image_tensor::<InnerB>(&prepared, config.image_size, &device);
            let targets =
                encode_targets::<InnerB>(&prepared, config.image_size, grid_size, &device);

            let output = valid_model.forward(images);
            let losses = detection_loss(&output, &targets, 1.0);

            let cls_val: f32 = losses.cls_loss.into_scalar().elem();
            let reg_val: f32 = losses.reg_loss.into_scalar().elem();
            valid_cls += cls_val as f64;
            valid_reg += reg_val as f64;
            valid_batches += 1;
        }

        let avg_valid_cls = valid_cls / valid_batches as f64;
        let avg_valid_reg = valid_reg / valid_batches as f64;

        println!(
            "Epoch {}/{} | Train cls={:.4} reg={:.4} total={:.4} | Valid cls={:.4} reg={:.4} total={:.4}",
            epoch + 1,
            config.epochs,
            avg_train_cls,
            avg_train_reg,
            avg_train_cls + avg_train_reg,
            avg_valid_cls,
            avg_valid_reg,
            avg_valid_cls + avg_valid_reg,
        );
    }

    println!("Treinamento concluído.");
    Ok(())
}
