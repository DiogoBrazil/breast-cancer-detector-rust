use burn::tensor::{Tensor, backend::Backend};

use super::target::GridTargets;
use crate::model::DetectionOutput;

pub struct LossOutput<B: Backend> {
    pub total: Tensor<B, 1>,
    pub cls_loss: Tensor<B, 1>,
    pub reg_loss: Tensor<B, 1>,
}

/// BCE with logits: -[y * log(σ(x)) + (1-y) * log(1-σ(x))]
/// Numericamente estável usando: max(x,0) - x*y + log(1 + exp(-|x|))
fn bce_with_logits<B: Backend>(logits: Tensor<B, 4>, targets: Tensor<B, 4>) -> Tensor<B, 1> {
    let abs_logits = logits.clone().abs();
    let relu_logits = logits.clone().clamp_min(0.0);

    // relu(x) - x*y + log(1 + exp(-|x|))
    let loss = relu_logits - logits * targets + (abs_logits.neg().exp() + 1.0).log();

    loss.mean()
}

/// Smooth L1 loss aplicada apenas nas posições positivas (onde há objeto).
fn smooth_l1_masked<B: Backend>(
    pred: Tensor<B, 4>,
    target: Tensor<B, 4>,
    mask: Tensor<B, 4>,
) -> Tensor<B, 1> {
    let diff = (pred - target).abs();

    // smooth_l1: 0.5*x² se |x|<1, |x|-0.5 caso contrário
    let ones = diff.clone().ones_like();
    let lt_one = diff.clone().lower(ones);
    let lt_one_f = lt_one.float();

    let quadratic = diff.clone().powf_scalar(2.0) * 0.5;
    let linear = diff - 0.5;

    let loss = lt_one_f.clone() * quadratic + (lt_one_f.neg() + 1.0) * linear;

    // Expandir máscara [B,1,H,W] -> [B,4,H,W] para casar com loss
    let mask4 = mask.clone().repeat_dim(1, 4);
    let masked_loss = loss * mask4;

    // Dividir pelo número de positivos (mask=1), clamp para evitar div/0
    let num_pos = mask.sum().clamp_min(1.0);
    masked_loss.sum() / num_pos
}

/// Calcula a loss total do detector.
/// cls_loss: BCE with logits sobre todo o grid.
/// reg_loss: Smooth L1 apenas nas posições positivas.
pub fn detection_loss<B: Backend>(
    output: &DetectionOutput<B>,
    targets: &GridTargets<B>,
    reg_weight: f32,
) -> LossOutput<B> {
    let cls_loss = bce_with_logits(output.cls_logits.clone(), targets.cls_target.clone());

    let reg_loss = smooth_l1_masked(
        output.reg_pred.clone(),
        targets.reg_target.clone(),
        targets.pos_mask.clone(),
    );

    let total = cls_loss.clone() + reg_loss.clone() * reg_weight;

    LossOutput {
        total,
        cls_loss,
        reg_loss,
    }
}
