use burn::tensor::{Tensor, backend::Backend};

use super::target::GridTargets;
use crate::model::DetectionOutput;

pub struct LossOutput<B: Backend> {
    pub total: Tensor<B, 1>,
    pub cls_loss: Tensor<B, 1>,
    pub reg_loss: Tensor<B, 1>,
}

/// Focal Loss with logits (RetinaNet, α=0.25, γ=2.0).
/// Reduz o peso dos negativos fáceis, crucial para grids com poucos positivos.
/// Numericamente estável via log-sigmoid.
fn focal_loss_with_logits<B: Backend>(logits: Tensor<B, 4>, targets: Tensor<B, 4>) -> Tensor<B, 1> {
    let alpha = 0.25f32;
    let gamma = 2.0f32;

    // p = sigmoid(logits), calculado de forma estável
    let p = burn::tensor::activation::sigmoid(logits.clone());

    // log(p) estável = logits - relu(logits) - log(1 + exp(-|logits|))
    // log(1-p) estável = -relu(logits) - log(1 + exp(-|logits|))
    let abs_logits = logits.clone().abs();
    let log_stable = (abs_logits.neg().exp() + 1.0).log(); // log(1 + exp(-|x|))
    let log_p = logits.clone() - logits.clone().clamp_min(0.0) - log_stable.clone();
    let log_1_minus_p = logits.clamp_min(0.0).neg() - log_stable;

    let num_pos = targets.clone().sum().clamp_min(1.0);

    // Focal weight e loss por elemento
    let p_t = targets.clone() * p.clone() + (targets.clone().neg() + 1.0) * (p.clone().neg() + 1.0);
    let focal_weight = (p_t.neg() + 1.0).powf_scalar(gamma);
    let alpha_t = targets.clone() * alpha + (targets.clone().neg() + 1.0) * (1.0 - alpha);

    let ce = targets.clone().neg() * log_p + (targets.neg() + 1.0).neg() * log_1_minus_p;

    let loss = alpha_t * focal_weight * ce;

    loss.sum() / num_pos
}

/// Smooth L1 loss aplicada apenas nas posições positivas (onde há objeto).
#[allow(dead_code)]
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

/// GIoU loss (Generalized IoU) aplicada apenas nas posições positivas.
/// Converte pred/target ltrb → xyxy usando coordenadas do grid, depois calcula GIoU.
/// Retorna mean(1 - GIoU) sobre os positivos.
fn giou_loss_masked<B: Backend>(
    pred: Tensor<B, 4>,   // [B, 4, H, W] — ltrb distances (já exp(), sempre > 0)
    target: Tensor<B, 4>, // [B, 4, H, W] — ltrb distances (GT)
    mask: Tensor<B, 4>,   // [B, 1, H, W] — positivos
    stride: f32,
) -> Tensor<B, 1> {
    let [_b, _, h, w] = pred.dims();
    let device = pred.device();

    // Construir grid de coordenadas cx[1,1,H,W] e cy[1,1,H,W]
    let mut cx_data = vec![0.0f32; h * w];
    let mut cy_data = vec![0.0f32; h * w];
    for gy in 0..h {
        for gx in 0..w {
            cx_data[gy * w + gx] = gx as f32 * stride + stride / 2.0;
            cy_data[gy * w + gx] = gy as f32 * stride + stride / 2.0;
        }
    }
    let cx = Tensor::<B, 1>::from_floats(cx_data.as_slice(), &device).reshape([1, 1, h, w]);
    let cy = Tensor::<B, 1>::from_floats(cy_data.as_slice(), &device).reshape([1, 1, h, w]);

    // Extrair canais ltrb: [B,4,H,W] → [B,1,H,W] via narrow no dim 1
    let pred_l = pred.clone().narrow(1, 0, 1);
    let pred_t = pred.clone().narrow(1, 1, 1);
    let pred_r = pred.clone().narrow(1, 2, 1);
    let pred_b = pred.narrow(1, 3, 1);

    let gt_l = target.clone().narrow(1, 0, 1);
    let gt_t = target.clone().narrow(1, 1, 1);
    let gt_r = target.clone().narrow(1, 2, 1);
    let gt_b = target.narrow(1, 3, 1);

    // Converter ltrb → xyxy (em pixel space)
    let pred_x1 = cx.clone() - pred_l * stride;
    let pred_y1 = cy.clone() - pred_t * stride;
    let pred_x2 = cx.clone() + pred_r * stride;
    let pred_y2 = cy.clone() + pred_b * stride;

    let gt_x1 = cx.clone() - gt_l * stride;
    let gt_y1 = cy.clone() - gt_t * stride;
    let gt_x2 = cx.clone() + gt_r * stride;
    let gt_y2 = cy + gt_b * stride;

    // Áreas
    let pred_area = (pred_x2.clone() - pred_x1.clone()) * (pred_y2.clone() - pred_y1.clone());
    let gt_area = (gt_x2.clone() - gt_x1.clone()) * (gt_y2.clone() - gt_y1.clone());

    // Interseção
    let inter_x1 = pred_x1.clone().max_pair(gt_x1.clone());
    let inter_y1 = pred_y1.clone().max_pair(gt_y1.clone());
    let inter_x2 = pred_x2.clone().min_pair(gt_x2.clone());
    let inter_y2 = pred_y2.clone().min_pair(gt_y2.clone());

    let inter_w = (inter_x2 - inter_x1).clamp_min(0.0);
    let inter_h = (inter_y2 - inter_y1).clamp_min(0.0);
    let inter_area = inter_w * inter_h;

    // União
    let union_area = (pred_area + gt_area - inter_area.clone()).clamp_min(1e-6);

    // IoU
    let iou = inter_area / union_area.clone();

    // Enclosing box (menor box que contém pred e gt)
    let encl_x1 = pred_x1.min_pair(gt_x1);
    let encl_y1 = pred_y1.min_pair(gt_y1);
    let encl_x2 = pred_x2.max_pair(gt_x2);
    let encl_y2 = pred_y2.max_pair(gt_y2);
    let encl_area = ((encl_x2 - encl_x1) * (encl_y2 - encl_y1)).clamp_min(1e-6);

    // GIoU = IoU - (encl_area - union_area) / encl_area
    let giou = iou - (encl_area.clone() - union_area) / encl_area;

    // Loss = (1 - GIoU) * mask, normalizada por num_positivos
    let loss = (giou.neg() + 1.0) * mask.clone();
    let num_pos = mask.sum().clamp_min(1.0);

    loss.sum() / num_pos
}

/// Calcula a loss total do detector.
/// cls_loss: Focal Loss sobre todo o grid.
/// reg_loss: GIoU loss apenas nas posições positivas.
pub fn detection_loss<B: Backend>(
    output: &DetectionOutput<B>,
    targets: &GridTargets<B>,
    reg_weight: f32,
    stride: f32,
) -> LossOutput<B> {
    let cls_loss = focal_loss_with_logits(output.cls_logits.clone(), targets.cls_target.clone());

    let reg_loss = giou_loss_masked(
        output.reg_pred.clone(),
        targets.reg_target.clone(),
        targets.pos_mask.clone(),
        stride,
    );

    let total = cls_loss.clone() + reg_loss.clone() * reg_weight;

    LossOutput {
        total,
        cls_loss,
        reg_loss,
    }
}
