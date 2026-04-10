#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub dataset_root: String,
    pub image_size: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub num_classes: usize,
    pub max_boxes: usize,
    pub patience: usize,
    pub warmup_epochs: usize,
    pub output_dir: String,
    /// "auto" para download automático, path para arquivo .pth, ou None para treinar do zero.
    pub pretrained_weights: Option<String>,
    /// Número de epochs com backbone congelado (só treina a head).
    pub freeze_backbone_epochs: usize,
    /// Fator de LR para o backbone durante fine-tuning (backbone_lr = base_lr × factor).
    pub backbone_lr_factor: f64,
    /// Aplicar data augmentation durante treino (flip horizontal, etc.).
    pub augment: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            dataset_root: "CBIS-DDSM.v1-cbis-ddsm.coco".to_string(),
            image_size: 416,
            batch_size: 12,
            epochs: 50,
            learning_rate: 1e-4,
            weight_decay: 1e-4,
            num_classes: 1,
            max_boxes: 16,
            patience: 8,
            warmup_epochs: 5,
            output_dir: "outputs".to_string(),
            pretrained_weights: Some("auto".to_string()),
            freeze_backbone_epochs: 5,
            backbone_lr_factor: 0.1,
            augment: true,
        }
    }
}
