use std::path::Path;

use anyhow::Result;
use image::{Rgb, RgbImage};

use crate::data::annotation::BoundingBox;
use crate::data::image_ops::{load_grayscale_image, preprocess_image_and_boxes};

fn draw_rect(img: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32, color: Rgb<u8>) {
    let w = img.width();
    let h = img.height();

    let x1 = x1.min(w.saturating_sub(1));
    let y1 = y1.min(h.saturating_sub(1));
    let x2 = x2.min(w.saturating_sub(1));
    let y2 = y2.min(h.saturating_sub(1));

    if x2 <= x1 || y2 <= y1 {
        return;
    }

    for x in x1..=x2 {
        img.put_pixel(x, y1, color);
        img.put_pixel(x, y2, color);
    }

    for y in y1..=y2 {
        img.put_pixel(x1, y, color);
        img.put_pixel(x2, y, color);
    }
}

pub fn save_preprocessed_sample_with_boxes(
    image_path: &Path,
    boxes: &[BoundingBox],
    image_size: usize,
    output_path: &Path,
) -> Result<()> {
    let gray = load_grayscale_image(image_path)?;
    let processed = preprocess_image_and_boxes(&gray, boxes, image_size)?;

    let mut rgb = RgbImage::new(image_size as u32, image_size as u32);

    for y in 0..image_size {
        for x in 0..image_size {
            let idx = y * image_size + x;
            let v = (processed.image_chw[idx] * 255.0).round().clamp(0.0, 255.0) as u8;
            rgb.put_pixel(x as u32, y as u32, Rgb([v, v, v]));
        }
    }

    for b in processed.boxes {
        draw_rect(
            &mut rgb,
            b[0].round() as u32,
            b[1].round() as u32,
            b[2].round() as u32,
            b[3].round() as u32,
            Rgb([255, 0, 0]),
        );
    }

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    rgb.save(output_path)?;
    Ok(())
}
