use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
    pub class_id: usize,
}

impl BoundingBox {
    pub fn width(&self) -> f32 {
        (self.x_max - self.x_min).max(0.0)
    }

    pub fn height(&self) -> f32 {
        (self.y_max - self.y_min).max(0.0)
    }

    pub fn is_valid(&self) -> bool {
        self.x_max > self.x_min && self.y_max > self.y_min
    }

    pub fn clamp(&mut self, max_w: f32, max_h: f32) {
        self.x_min = self.x_min.clamp(0.0, max_w);
        self.y_min = self.y_min.clamp(0.0, max_h);
        self.x_max = self.x_max.clamp(0.0, max_w);
        self.y_max = self.y_max.clamp(0.0, max_h);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationFile {
    pub image: String,
    pub width: u32,
    pub height: u32,
    pub boxes: Vec<BoundingBox>,
    pub class_name: String,
}
