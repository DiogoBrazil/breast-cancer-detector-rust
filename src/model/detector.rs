use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend};

use super::backbone::Backbone;
use super::head::{DetectionHead, DetectionOutput};

/// Detector anchor-free: backbone convolucional + detection head (cls + reg).
///
/// Input:  imagens [B, 3, 416, 416]
/// Output: cls_logits [B, 1, 26, 26] + reg_pred [B, 4, 26, 26]
#[derive(Module, Debug)]
pub struct SimpleDetector<B: Backend> {
    backbone: Backbone<B>,
    head: DetectionHead<B>,
}

impl<B: Backend> SimpleDetector<B> {
    pub fn new(device: &B::Device, num_classes: usize) -> Self {
        let backbone = Backbone::new(device);
        let head = DetectionHead::new(device, backbone.out_channels(), num_classes);
        Self { backbone, head }
    }

    pub fn forward(&self, images: Tensor<B, 4>) -> DetectionOutput<B> {
        let features = self.backbone.forward(images);
        self.head.forward(features)
    }
}
