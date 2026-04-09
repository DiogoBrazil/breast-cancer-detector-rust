use std::path::Path;

use anyhow::{Context, Result};
use image::{Rgb, RgbImage};

use crate::data::annotation::BoundingBox;
use crate::data::image_ops::{load_grayscale_image, preprocess_image_and_boxes};
use crate::inference::detect::Detection;

pub(crate) fn draw_rect(img: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32, color: Rgb<u8>, thickness: u32) {
    let w = img.width();
    let h = img.height();

    let x1 = x1.min(w.saturating_sub(1));
    let y1 = y1.min(h.saturating_sub(1));
    let x2 = x2.min(w.saturating_sub(1));
    let y2 = y2.min(h.saturating_sub(1));

    if x2 <= x1 || y2 <= y1 {
        return;
    }

    // Top and bottom edges
    for x in x1..=x2 {
        for t in 0..thickness {
            let yt = y1 + t;
            let yb = y2.saturating_sub(t);
            if yt < h {
                img.put_pixel(x, yt, color);
            }
            if yb < h {
                img.put_pixel(x, yb, color);
            }
        }
    }

    // Left and right edges
    for y in y1..=y2 {
        for t in 0..thickness {
            let xl = x1 + t;
            let xr = x2.saturating_sub(t);
            if xl < w {
                img.put_pixel(xl, y, color);
            }
            if xr < w {
                img.put_pixel(xr, y, color);
            }
        }
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
            2,
        );
    }

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    rgb.save(output_path)?;
    Ok(())
}

/// 5x7 bitmap glyphs for digits 0-9, '.', and '%'.
const GLYPH_W: u32 = 5;
const GLYPH_H: u32 = 7;

fn glyph(ch: char) -> Option<[u8; 7]> {
    Some(match ch {
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        '3' => [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100],
        '%' => [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011],
        _ => return None,
    })
}

fn draw_label(img: &mut RgbImage, text: &str, x: u32, y: u32, scale: u32, fg: Rgb<u8>, bg: Rgb<u8>) {
    let w = img.width();
    let h = img.height();
    let char_w = GLYPH_W * scale + scale; // glyph + 1*scale spacing
    let label_w = text.len() as u32 * char_w + scale;
    let label_h = GLYPH_H * scale + 2 * scale;

    // Draw background rectangle
    let y_bg = y.saturating_sub(label_h);
    for py in y_bg..y.min(h) {
        for px in x..(x + label_w).min(w) {
            img.put_pixel(px, py, bg);
        }
    }

    // Draw each character
    let text_y = y.saturating_sub(label_h) + scale;
    for (ci, ch) in text.chars().enumerate() {
        if let Some(rows) = glyph(ch) {
            let cx = x + scale + ci as u32 * char_w;
            for (row_idx, &row) in rows.iter().enumerate() {
                for bit in 0..GLYPH_W {
                    if row & (1 << (GLYPH_W - 1 - bit)) != 0 {
                        for sy in 0..scale {
                            for sx in 0..scale {
                                let px = cx + bit * scale + sx;
                                let py = text_y + row_idx as u32 * scale + sy;
                                if px < w && py < h {
                                    img.put_pixel(px, py, fg);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn save_image_with_detections(
    image_path: &Path,
    detections: &[Detection],
    output_path: &Path,
) -> Result<()> {
    let img = image::ImageReader::open(image_path)
        .with_context(|| format!("Failed to open image {}", image_path.display()))?
        .decode()
        .with_context(|| format!("Failed to decode image {}", image_path.display()))?;

    let mut rgb = img.to_rgb8();
    let green = Rgb([0, 255, 0]);

    let img_max = rgb.width().max(rgb.height());
    let thickness = (img_max / 300).max(2);
    let font_scale = (img_max / 500).max(2);

    for det in detections {
        let x1 = det.x_min.max(0.0).round() as u32;
        let y1 = det.y_min.max(0.0).round() as u32;
        let x2 = det.x_max.max(0.0).round() as u32;
        let y2 = det.y_max.max(0.0).round() as u32;

        draw_rect(&mut rgb, x1, y1, x2, y2, green, thickness);

        let pct = format!("{:.1}%", det.confidence * 100.0);
        draw_label(&mut rgb, &pct, x1, y1, font_scale, Rgb([0, 0, 0]), green);
    }

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    rgb.save(output_path)?;
    println!(
        "{} detection(s) drawn on {}",
        detections.len(),
        output_path.display()
    );
    Ok(())
}
