use anyhow::{Context, Result};
use image::{DynamicImage, GrayImage, ImageReader, Luma, imageops::FilterType};
use rand::Rng;

use super::annotation::BoundingBox;

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

/// Aplica augmentações aleatórias a uma imagem e seus boxes.
/// Retorna a imagem (possivelmente transformada) e boxes correspondentes.
pub fn augment(
    image: &GrayImage,
    boxes: &[BoundingBox],
    rng: &mut impl Rng,
) -> (GrayImage, Vec<BoundingBox>) {
    let mut img = image.clone();
    let mut bxs = boxes.to_vec();

    // Horizontal flip: 50% de probabilidade
    if rng.gen_bool(0.5) {
        let (fi, fb) = flip_horizontal(&img, &bxs);
        img = fi;
        bxs = fb;
    }

    (img, bxs)
}
