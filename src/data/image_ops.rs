use anyhow::{Context, Result};
use image::{DynamicImage, GrayImage, ImageReader, Luma, imageops::FilterType};
use rand::Rng;

use super::annotation::{BoundingBox, MIN_BOX_AREA};
use crate::training::config::AugmentConfig;

#[derive(Debug, Clone)]
pub struct ProcessedSample {
    pub image_chw: Vec<f32>,  // [3, H, W]
    pub boxes: Vec<[f32; 4]>, // xyxy já ajustado
    pub labels: Vec<usize>,
    pub image_size: usize,
    pub orig_width: u32,
    pub orig_height: u32,
    pub resized_width: u32,
    pub resized_height: u32,
    pub pad_x: u32,
    pub pad_y: u32,
    pub scale: f32,
}

pub fn load_grayscale_image(path: &std::path::Path) -> Result<GrayImage> {
    let img = ImageReader::open(path)
        .with_context(|| format!("Failed to open image {}", path.display()))?
        .decode()
        .with_context(|| format!("Failed to decode image {}", path.display()))?;

    Ok(to_grayscale(img))
}

fn to_grayscale(img: DynamicImage) -> GrayImage {
    img.to_luma8()
}

pub fn preprocess_image_and_boxes(
    image: &GrayImage,
    boxes: &[BoundingBox],
    target_size: usize,
) -> Result<ProcessedSample> {
    let orig_width = image.width();
    let orig_height = image.height();

    let target_size_u32 = target_size as u32;

    let scale_w = target_size_u32 as f32 / orig_width as f32;
    let scale_h = target_size_u32 as f32 / orig_height as f32;
    let scale = scale_w.min(scale_h);

    let resized_width = ((orig_width as f32 * scale).round() as u32).max(1);
    let resized_height = ((orig_height as f32 * scale).round() as u32).max(1);

    let resized =
        image::imageops::resize(image, resized_width, resized_height, FilterType::Triangle);

    let mut canvas = GrayImage::new(target_size_u32, target_size_u32);

    // Fundo preto
    for y in 0..target_size_u32 {
        for x in 0..target_size_u32 {
            canvas.put_pixel(x, y, Luma([0u8]));
        }
    }

    let pad_x = (target_size_u32 - resized_width) / 2;
    let pad_y = (target_size_u32 - resized_height) / 2;

    image::imageops::overlay(&mut canvas, &resized, pad_x as i64, pad_y as i64);

    let mut adj_boxes = Vec::new();
    let mut labels = Vec::new();

    for b in boxes {
        let mut x_min = b.x_min * scale + pad_x as f32;
        let mut y_min = b.y_min * scale + pad_y as f32;
        let mut x_max = b.x_max * scale + pad_x as f32;
        let mut y_max = b.y_max * scale + pad_y as f32;

        x_min = x_min.clamp(0.0, target_size_u32 as f32);
        y_min = y_min.clamp(0.0, target_size_u32 as f32);
        x_max = x_max.clamp(0.0, target_size_u32 as f32);
        y_max = y_max.clamp(0.0, target_size_u32 as f32);

        if x_max > x_min && y_max > y_min {
            adj_boxes.push([x_min, y_min, x_max, y_max]);
            labels.push(b.class_id);
        }
    }

    // Gray -> 3 canais
    let mut chw = vec![0.0f32; 3 * target_size * target_size];

    for y in 0..target_size {
        for x in 0..target_size {
            let pixel = canvas.get_pixel(x as u32, y as u32)[0] as f32 / 255.0;

            let idx_hw = y * target_size + x;
            chw[idx_hw] = pixel; // canal 0
            chw[target_size * target_size + idx_hw] = pixel; // canal 1
            chw[2 * target_size * target_size + idx_hw] = pixel; // canal 2
        }
    }

    Ok(ProcessedSample {
        image_chw: chw,
        boxes: adj_boxes,
        labels,
        image_size: target_size,
        orig_width,
        orig_height,
        resized_width,
        resized_height,
        pad_x,
        pad_y,
        scale,
    })
}

/// Espelha imagem e bounding boxes horizontalmente.
pub fn flip_horizontal(image: &GrayImage, boxes: &[BoundingBox]) -> (GrayImage, Vec<BoundingBox>) {
    let flipped = image::imageops::flip_horizontal(image);
    let w = image.width() as f32;

    let flipped_boxes = boxes
        .iter()
        .map(|b| BoundingBox {
            x_min: w - b.x_max,
            y_min: b.y_min,
            x_max: w - b.x_min,
            y_max: b.y_max,
            class_id: b.class_id,
        })
        .collect();

    (flipped, flipped_boxes)
}

/// Ajusta brilho da imagem multiplicando cada pixel por um fator.
fn adjust_brightness(image: &GrayImage, factor: f32) -> GrayImage {
    let mut out = image.clone();
    for pixel in out.pixels_mut() {
        pixel[0] = (pixel[0] as f32 * factor).clamp(0.0, 255.0) as u8;
    }
    out
}

/// Ajusta contraste da imagem em relação à média dos pixels.
fn adjust_contrast(image: &GrayImage, factor: f32) -> GrayImage {
    let mean: f32 = image.pixels().map(|p| p[0] as f32).sum::<f32>()
        / (image.width() * image.height()) as f32;

    let mut out = image.clone();
    for pixel in out.pixels_mut() {
        pixel[0] = ((pixel[0] as f32 - mean) * factor + mean).clamp(0.0, 255.0) as u8;
    }
    out
}

/// Adiciona ruído gaussiano usando Box-Muller transform.
fn add_gaussian_noise(image: &GrayImage, sigma: f32, rng: &mut impl Rng) -> GrayImage {
    let mut out = image.clone();
    for pixel in out.pixels_mut() {
        let u1: f32 = rng.r#gen::<f32>().max(1e-10);
        let u2: f32 = rng.r#gen::<f32>();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        pixel[0] = (pixel[0] as f32 + z * sigma).clamp(0.0, 255.0) as u8;
    }
    out
}

/// Rotaciona imagem e bounding boxes ao redor do centro.
/// Usa inverse mapping com interpolação nearest-neighbor.
fn rotate_image_and_boxes(
    image: &GrayImage,
    angle_deg: f32,
    boxes: &[BoundingBox],
) -> (GrayImage, Vec<BoundingBox>) {
    let w = image.width() as f32;
    let h = image.height() as f32;
    let cx = w / 2.0;
    let cy = h / 2.0;

    let angle_rad = angle_deg * std::f32::consts::PI / 180.0;
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    // Inverse mapping: para cada pixel destino, calcular coordenada fonte
    let mut out = GrayImage::new(image.width(), image.height());
    for dy in 0..image.height() {
        for dx in 0..image.width() {
            let rx = dx as f32 - cx;
            let ry = dy as f32 - cy;
            // Rotação inversa (-angle)
            let sx = rx * cos_a + ry * sin_a + cx;
            let sy = -rx * sin_a + ry * cos_a + cy;

            let sxi = sx.round() as i32;
            let syi = sy.round() as i32;

            if sxi >= 0 && sxi < w as i32 && syi >= 0 && syi < h as i32 {
                out.put_pixel(dx, dy, *image.get_pixel(sxi as u32, syi as u32));
            }
            // Pixels fora ficam pretos (0)
        }
    }

    // Rotacionar boxes: transformar 4 cantos, pegar AABB
    let rotated_boxes: Vec<BoundingBox> = boxes
        .iter()
        .filter_map(|b| {
            let corners = [
                (b.x_min, b.y_min),
                (b.x_max, b.y_min),
                (b.x_max, b.y_max),
                (b.x_min, b.y_max),
            ];

            let mut min_x = f32::MAX;
            let mut min_y = f32::MAX;
            let mut max_x = f32::MIN;
            let mut max_y = f32::MIN;

            for (px, py) in &corners {
                let rx = px - cx;
                let ry = py - cy;
                let nx = rx * cos_a - ry * sin_a + cx;
                let ny = rx * sin_a + ry * cos_a + cy;
                min_x = min_x.min(nx);
                min_y = min_y.min(ny);
                max_x = max_x.max(nx);
                max_y = max_y.max(ny);
            }

            // Clampar aos limites da imagem
            min_x = min_x.clamp(0.0, w);
            min_y = min_y.clamp(0.0, h);
            max_x = max_x.clamp(0.0, w);
            max_y = max_y.clamp(0.0, h);

            let new_box = BoundingBox {
                x_min: min_x,
                y_min: min_y,
                x_max: max_x,
                y_max: max_y,
                class_id: b.class_id,
            };

            if new_box.area() >= MIN_BOX_AREA {
                Some(new_box)
            } else {
                None
            }
        })
        .collect();

    (out, rotated_boxes)
}

/// Random scale/crop: zoom in (scale > 1) ou zoom out (scale < 1).
fn random_scale_crop(
    image: &GrayImage,
    boxes: &[BoundingBox],
    scale: f32,
    rng: &mut impl Rng,
) -> (GrayImage, Vec<BoundingBox>) {
    let w = image.width();
    let h = image.height();

    if scale >= 1.0 {
        // Zoom in: crop uma região menor e resize up
        let crop_w = (w as f32 / scale).round() as u32;
        let crop_h = (h as f32 / scale).round() as u32;

        let max_ox = w.saturating_sub(crop_w);
        let max_oy = h.saturating_sub(crop_h);
        let ox = if max_ox > 0 { rng.gen_range(0..=max_ox) } else { 0 };
        let oy = if max_oy > 0 { rng.gen_range(0..=max_oy) } else { 0 };

        let cropped = image::imageops::crop_imm(image, ox, oy, crop_w, crop_h).to_image();
        let resized = image::imageops::resize(&cropped, w, h, FilterType::Triangle);

        let adj_boxes: Vec<BoundingBox> = boxes
            .iter()
            .filter_map(|b| {
                let new_box = BoundingBox {
                    x_min: ((b.x_min - ox as f32) * scale).clamp(0.0, w as f32),
                    y_min: ((b.y_min - oy as f32) * scale).clamp(0.0, h as f32),
                    x_max: ((b.x_max - ox as f32) * scale).clamp(0.0, w as f32),
                    y_max: ((b.y_max - oy as f32) * scale).clamp(0.0, h as f32),
                    class_id: b.class_id,
                };
                if new_box.area() >= MIN_BOX_AREA { Some(new_box) } else { None }
            })
            .collect();

        (resized, adj_boxes)
    } else {
        // Zoom out: resize menor e colocar em canvas preto
        let new_w = (w as f32 * scale).round() as u32;
        let new_h = (h as f32 * scale).round() as u32;

        let resized = image::imageops::resize(image, new_w, new_h, FilterType::Triangle);

        let mut canvas = GrayImage::new(w, h);
        let max_ox = w.saturating_sub(new_w);
        let max_oy = h.saturating_sub(new_h);
        let ox = if max_ox > 0 { rng.gen_range(0..=max_ox) } else { 0 };
        let oy = if max_oy > 0 { rng.gen_range(0..=max_oy) } else { 0 };

        image::imageops::overlay(&mut canvas, &resized, ox as i64, oy as i64);

        let adj_boxes: Vec<BoundingBox> = boxes
            .iter()
            .filter_map(|b| {
                let new_box = BoundingBox {
                    x_min: (b.x_min * scale + ox as f32).clamp(0.0, w as f32),
                    y_min: (b.y_min * scale + oy as f32).clamp(0.0, h as f32),
                    x_max: (b.x_max * scale + ox as f32).clamp(0.0, w as f32),
                    y_max: (b.y_max * scale + oy as f32).clamp(0.0, h as f32),
                    class_id: b.class_id,
                };
                if new_box.area() >= MIN_BOX_AREA { Some(new_box) } else { None }
            })
            .collect();

        (canvas, adj_boxes)
    }
}

/// Aplica augmentações aleatórias a uma imagem e seus boxes.
/// Ordem: flip → rotação → scale/crop → brightness → contrast → noise
pub fn augment(
    image: &GrayImage,
    boxes: &[BoundingBox],
    rng: &mut impl Rng,
    cfg: &AugmentConfig,
) -> (GrayImage, Vec<BoundingBox>) {
    let mut img = image.clone();
    let mut bxs = boxes.to_vec();

    // 1. Horizontal flip
    if rng.gen_bool(cfg.flip_prob) {
        let (fi, fb) = flip_horizontal(&img, &bxs);
        img = fi;
        bxs = fb;
    }

    // 2. Random rotation
    if rng.gen_bool(cfg.rotation_prob) {
        let angle = rng.gen_range(-cfg.rotation_max_deg..=cfg.rotation_max_deg);
        let (ri, rb) = rotate_image_and_boxes(&img, angle, &bxs);
        if !rb.is_empty() {
            img = ri;
            bxs = rb;
        }
    }

    // 3. Random scale/crop
    if rng.gen_bool(cfg.scale_prob) {
        let scale = rng.gen_range(cfg.scale_range.0..=cfg.scale_range.1);
        let (si, sb) = random_scale_crop(&img, &bxs, scale, rng);
        if !sb.is_empty() {
            img = si;
            bxs = sb;
        }
    }

    // 4. Random brightness
    if rng.gen_bool(cfg.brightness_prob) {
        let factor = rng.gen_range(cfg.brightness_range.0..=cfg.brightness_range.1);
        img = adjust_brightness(&img, factor);
    }

    // 5. Random contrast
    if rng.gen_bool(cfg.contrast_prob) {
        let factor = rng.gen_range(cfg.contrast_range.0..=cfg.contrast_range.1);
        img = adjust_contrast(&img, factor);
    }

    // 6. Gaussian noise
    if rng.gen_bool(cfg.noise_prob) {
        let sigma = rng.gen_range(cfg.noise_sigma_range.0..=cfg.noise_sigma_range.1);
        img = add_gaussian_noise(&img, sigma, rng);
    }

    (img, bxs)
}
