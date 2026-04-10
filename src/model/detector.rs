use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend};

use super::head::{DetectionHead, DetectionOutput};
use super::resnet::ResNetBackbone;

/// Detector anchor-free: ResNet-18 backbone (até layer3) + detection head (cls + reg).
///
/// Input:  imagens [B, 3, 416, 416] (pixels [0, 1])
/// Output: cls_logits [B, 1, 26, 26] + reg_pred [B, 4, 26, 26]
#[derive(Module, Debug)]
pub struct Detector<B: Backend> {
    pub backbone: ResNetBackbone<B>,
    pub head: DetectionHead<B>,
}

impl<B: Backend> Detector<B> {
    pub fn new(device: &B::Device, num_classes: usize) -> Self {
        let backbone = ResNetBackbone::new(device);
        let head = DetectionHead::new(device, backbone.out_channels(), num_classes);
        Self { backbone, head }
    }

    pub fn forward(&self, images: Tensor<B, 4>) -> DetectionOutput<B> {
        let features = self.backbone.forward(images);
        self.head.forward(features)
    }
}
