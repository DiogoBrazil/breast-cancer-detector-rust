/// Configuração de data augmentation com probabilidades e ranges.
#[derive(Debug, Clone)]
pub struct AugmentConfig {
    pub flip_prob: f64,
    pub rotation_prob: f64,
    pub rotation_max_deg: f32,
    pub scale_prob: f64,
    pub scale_range: (f32, f32),
    pub brightness_prob: f64,
    pub brightness_range: (f32, f32),
    pub contrast_prob: f64,
    pub contrast_range: (f32, f32),
    pub noise_prob: f64,
    pub noise_sigma_range: (f32, f32),
}

impl Default for AugmentConfig {
    fn default() -> Self {
        Self {
            flip_prob: 0.5,
            rotation_prob: 0.4,
            rotation_max_deg: 15.0,
            scale_prob: 0.4,
            scale_range: (0.8, 1.2),
            brightness_prob: 0.5,
            brightness_range: (0.8, 1.2),
            contrast_prob: 0.5,
            contrast_range: (0.8, 1.2),
            noise_prob: 0.3,
            noise_sigma_range: (10.0, 25.0),
        }
    }
}

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
    /// Aplicar data augmentation durante treino.
    pub augment: bool,
    /// Configuração detalhada de augmentation.
    pub augment_config: AugmentConfig,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            dataset_root: "CBIS-DDSM.v1-cbis-ddsm.coco".to_string(),
            image_size: 416,
            batch_size: 16,
            epochs: 80,
            learning_rate: 1e-4,
            weight_decay: 5e-4,
            num_classes: 1,
            max_boxes: 16,
            patience: 8,
            warmup_epochs: 5,
            output_dir: "outputs".to_string(),
            pretrained_weights: Some("auto".to_string()),
            freeze_backbone_epochs: 5,
            backbone_lr_factor: 0.1,
            augment: true,
            augment_config: AugmentConfig::default(),
        }
    }
}
