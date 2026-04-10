use std::path::Path;

use anyhow::Result;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::Tensor;
use burn_wgpu::{Wgpu, WgpuDevice};

use crate::data::image_ops::{load_grayscale_image, preprocess_image_and_boxes};
use crate::model::Detector;
use crate::model::resnet::BACKBONE_STRIDE;

use super::draw::save_image_with_detections;

#[derive(Debug, Clone)]
pub struct Detection {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
    pub confidence: f32,
}

pub fn run_detection(
    image_path: &Path,
    model_path: &Path,
    output_path: &Path,
    image_size: usize,
    num_classes: usize,
    conf_threshold: f32,
    nms_threshold: f32,
) -> Result<Vec<Detection>> {
    let device = WgpuDevice::DefaultDevice;

    // 1. Preprocess image
    let gray = load_grayscale_image(image_path)?;
    let processed = preprocess_image_and_boxes(&gray, &[], image_size)?;

    // 2. Build input tensor [1, 3, H, W]
    let images = Tensor::<Wgpu, 1>::from_floats(processed.image_chw.as_slice(), &device)
        .reshape([1, 3, image_size, image_size]);

    // 3. Load model
    let model = Detector::<Wgpu>::new(&device, num_classes);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    let record = recorder
        .load(model_path.into(), &device)
        .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;
    let model = model.load_record(record);

    // 4. Forward pass
    let output = model.forward(images);

    // 5. Decode predictions
    let grid_size = image_size / BACKBONE_STRIDE;
    let stride = BACKBONE_STRIDE as f32;

    let cls_data: Vec<f32> = output.cls_logits.into_data().to_vec().unwrap();
    let reg_data: Vec<f32> = output.reg_pred.into_data().to_vec().unwrap();

    let mut detections = decode_grid_predictions(
        &cls_data,
        &reg_data,
        grid_size,
        stride,
        image_size as f32,
        conf_threshold,
    );

    // 6. NMS
    let detections = nms(&mut detections, nms_threshold);

    // 7. Map back to original image coords
    let detections = map_to_original_coords(
        detections,
        processed.pad_x as f32,
        processed.pad_y as f32,
        processed.scale,
        processed.orig_width as f32,
        processed.orig_height as f32,
    );

    // 8. Draw and save
    save_image_with_detections(image_path, &detections, output_path)?;

    Ok(detections)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn decode_grid_predictions(
    cls_data: &[f32],
    reg_data: &[f32],
    grid_size: usize,
    stride: f32,
    image_size: f32,
    conf_threshold: f32,
) -> Vec<Detection> {
    let mut detections = Vec::new();
    let grid_cells = grid_size * grid_size;

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            let idx = gy * grid_size + gx;
            let conf = sigmoid(cls_data[idx]);

            if conf < conf_threshold {
                continue;
            }

            let cx = gx as f32 * stride + stride / 2.0;
            let cy = gy as f32 * stride + stride / 2.0;

            let l = reg_data[idx];
            let t = reg_data[grid_cells + idx];
            let r = reg_data[2 * grid_cells + idx];
            let b = reg_data[3 * grid_cells + idx];

            let x_min = (cx - l * stride).clamp(0.0, image_size);
            let y_min = (cy - t * stride).clamp(0.0, image_size);
            let x_max = (cx + r * stride).clamp(0.0, image_size);
            let y_max = (cy + b * stride).clamp(0.0, image_size);

            if x_max > x_min && y_max > y_min {
                detections.push(Detection {
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    confidence: conf,
                });
            }
        }
    }

    detections
}

fn iou(a: &Detection, b: &Detection) -> f32 {
    let inter_x_min = a.x_min.max(b.x_min);
    let inter_y_min = a.y_min.max(b.y_min);
    let inter_x_max = a.x_max.min(b.x_max);
    let inter_y_max = a.y_max.min(b.y_max);

    let inter_area = (inter_x_max - inter_x_min).max(0.0) * (inter_y_max - inter_y_min).max(0.0);
    let area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min);
    let area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min);
    let union_area = area_a + area_b - inter_area;

    if union_area <= 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}

fn nms(detections: &mut [Detection], iou_threshold: f32) -> Vec<Detection> {
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut keep = Vec::new();
    let mut suppressed = vec![false; detections.len()];

    for i in 0..detections.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(detections[i].clone());

        for j in (i + 1)..detections.len() {
            if !suppressed[j] && iou(&detections[i], &detections[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}

fn map_to_original_coords(
    mut detections: Vec<Detection>,
    pad_x: f32,
    pad_y: f32,
    scale: f32,
    orig_w: f32,
    orig_h: f32,
) -> Vec<Detection> {
    for det in &mut detections {
        det.x_min = ((det.x_min - pad_x) / scale).clamp(0.0, orig_w);
        det.y_min = ((det.y_min - pad_y) / scale).clamp(0.0, orig_h);
        det.x_max = ((det.x_max - pad_x) / scale).clamp(0.0, orig_w);
        det.y_max = ((det.y_max - pad_y) / scale).clamp(0.0, orig_h);
    }
    detections
}
