use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use burn::backend::Autodiff;
use burn::data::dataset::Dataset;
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::grad_clipping::GradientClippingConfig;
use burn::prelude::ElementConversion;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::Tensor;
use burn_wgpu::{Wgpu, WgpuDevice};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;

use super::config::TrainConfig;
use super::loss::detection_loss;
use super::target::encode_targets;
use crate::data::batcher::{PreparedSample, prepare_sample};
use crate::data::dataset::MammogramDataset;
use crate::model::SimpleDetector;
use crate::model::backbone::BACKBONE_STRIDE;

/// Warmup linear + cosine decay.
fn compute_lr(epoch: usize, total_epochs: usize, base_lr: f64, warmup_epochs: usize) -> f64 {
    if epoch < warmup_epochs {
        base_lr * (epoch + 1) as f64 / warmup_epochs as f64
    } else {
        let min_lr = base_lr / 100.0;
        let progress =
            (epoch - warmup_epochs) as f64 / (total_epochs - warmup_epochs).max(1) as f64;
        min_lr + 0.5 * (base_lr - min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}

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
    let grid_size = config.image_size / BACKBONE_STRIDE;

    // Dataset
    let train_dataset = MammogramDataset::new(format!("{}/train", config.dataset_root))?;
    let valid_dataset = MammogramDataset::new(format!("{}/valid", config.dataset_root))?;
    println!(
        "Train: {} samples | Valid: {} samples",
        train_dataset.len(),
        valid_dataset.len()
    );

    // Pré-carregar todas as amostras em memória (evita I/O repetido a cada época)
    println!("Pré-carregando train dataset em memória...");
    let train_prepared: Vec<PreparedSample> = (0..train_dataset.len())
        .into_par_iter()
        .map(|idx| {
            let sample = train_dataset.get(idx).expect("sample should exist");
            prepare_sample(&sample, config.image_size)
        })
        .collect::<Result<Vec<_>>>()?;
    println!("  -> {} amostras de treino carregadas.", train_prepared.len());

    println!("Pré-carregando valid dataset em memória...");
    let valid_prepared: Vec<PreparedSample> = (0..valid_dataset.len())
        .into_par_iter()
        .map(|idx| {
            let sample = valid_dataset.get(idx).expect("sample should exist");
            prepare_sample(&sample, config.image_size)
        })
        .collect::<Result<Vec<_>>>()?;
    println!("  -> {} amostras de validação carregadas.", valid_prepared.len());

    // Criar diretório de saída
    std::fs::create_dir_all(&config.output_dir)?;

    // Modelo e optimizer
    let mut model = SimpleDetector::<B>::new(&device, config.num_classes);
    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(
            config.weight_decay as f32,
        )))
        .with_grad_clipping(Some(GradientClippingConfig::Norm(10.0)))
        .init();

    let mut rng = thread_rng();
    let mut best_valid_loss = f64::MAX;
    let mut epochs_without_improvement: usize = 0;

    // CSV log
    let csv_path = PathBuf::from(&config.output_dir).join("training_log.csv");
    let mut csv_file = std::fs::File::create(&csv_path)?;
    writeln!(csv_file, "epoch,train_cls_loss,train_reg_loss,train_total_loss,valid_cls_loss,valid_reg_loss,valid_total_loss,best_valid_loss,is_best,learning_rate,elapsed_secs")?;

    for epoch in 0..config.epochs {
        let epoch_start = Instant::now();
        let current_lr = compute_lr(epoch, config.epochs, config.learning_rate, config.warmup_epochs);

        // === TRAIN ===
        let mut indices: Vec<usize> = (0..train_dataset.len()).collect();
        indices.shuffle(&mut rng);

        let mut train_cls = 0.0f64;
        let mut train_reg = 0.0f64;
        let mut train_batches = 0usize;

        for batch_start in (0..indices.len()).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(indices.len());
            let batch_indices = &indices[batch_start..batch_end];

            // Buscar samples pré-carregados da memória
            let prepared: Vec<PreparedSample> = batch_indices
                .iter()
                .map(|&idx| train_prepared[idx].clone())
                .collect();

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
            model = optim.step(current_lr, model, grads);
        }

        let avg_train_cls = train_cls / train_batches as f64;
        let avg_train_reg = train_reg / train_batches as f64;

        // === VALIDATION ===
        let valid_model = model.valid();
        let mut valid_cls = 0.0f64;
        let mut valid_reg = 0.0f64;
        let mut valid_batches = 0usize;

        for batch_start in (0..valid_prepared.len()).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(valid_prepared.len());

            // Buscar samples pré-carregados da memória
            let prepared: Vec<PreparedSample> = (batch_start..batch_end)
                .map(|idx| valid_prepared[idx].clone())
                .collect();

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

        let valid_total = avg_valid_cls + avg_valid_reg;

        println!(
            "Epoch {}/{} | Train cls={:.4} reg={:.4} total={:.4} | Valid cls={:.4} reg={:.4} total={:.4}",
            epoch + 1,
            config.epochs,
            avg_train_cls,
            avg_train_reg,
            avg_train_cls + avg_train_reg,
            avg_valid_cls,
            avg_valid_reg,
            valid_total,
        );

        // Salvar melhor modelo (menor validation loss)
        let is_best = valid_total < best_valid_loss;
        if is_best {
            best_valid_loss = valid_total;
            epochs_without_improvement = 0;
            let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
            let best_path = PathBuf::from(&config.output_dir).join("model_best");
            recorder
                .record(model.clone().into_record(), best_path.clone())
                .map_err(|e| anyhow::anyhow!("Erro ao salvar melhor modelo: {}", e))?;
            println!(
                "  -> Melhor modelo salvo (valid_loss={:.4}) em {}.mpk",
                valid_total,
                best_path.display()
            );
        } else {
            epochs_without_improvement += 1;
        }

        // CSV append
        let elapsed = epoch_start.elapsed().as_secs_f64();
        writeln!(
            csv_file,
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.1}",
            epoch + 1,
            avg_train_cls,
            avg_train_reg,
            avg_train_cls + avg_train_reg,
            avg_valid_cls,
            avg_valid_reg,
            valid_total,
            best_valid_loss,
            is_best,
            current_lr,
            elapsed,
        )?;
        csv_file.flush()?;

        // Early stopping
        if epochs_without_improvement >= config.patience {
            println!(
                "Early stopping: sem melhora há {} épocas. Melhor valid_loss={:.4}",
                config.patience, best_valid_loss
            );
            break;
        }
    }

    // Salvar modelo final
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    let final_path = PathBuf::from(&config.output_dir).join("model_final");
    recorder
        .record(model.into_record(), final_path.clone())
        .map_err(|e| anyhow::anyhow!("Erro ao salvar modelo final: {}", e))?;
    println!("Modelo final salvo em {}.mpk", final_path.display());

    println!("Treinamento concluído.");
    Ok(())
}
