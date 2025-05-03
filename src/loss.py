import torch
from torch import cat, clip, ones, sum, tensor
from torch.nn import (BCEWithLogitsLoss, CosineSimilarity, CrossEntropyLoss)

from src.logits_computation import lookup_and_multiply

ce_loss = CrossEntropyLoss(reduction="none")
binary_ce_loss = BCEWithLogitsLoss(reduction="none")
cos_sim = CosineSimilarity(eps=1e-5, dim=-1)


def _elementwise_sampled_softmax_loss(pos_logits, neg_logits, mask, target):
    sm_logits = cat((pos_logits, neg_logits), dim=-1)
    shape = sm_logits.shape
    return (ce_loss(sm_logits.reshape([-1, shape[-1]]), target).reshape([shape[0], shape[1]]) * mask)


def sampled_softmax_loss(pos_logits, neg_logits, mask, device="cpu"):
    target = tensor([0], device=device).tile(mask.numel())
    elementwise_ssm_loss = _elementwise_sampled_softmax_loss(pos_logits, neg_logits, mask, target)
    return sum(elementwise_ssm_loss) / sum(mask), elementwise_ssm_loss


def binary_cross_entropy_loss(pos_logits, labels, mask, device="cpu"):
    elementwise_binary_ce_loss = binary_ce_loss(pos_logits, labels.unsqueeze(-1)) * mask.unsqueeze(-1)
    return elementwise_binary_ce_loss.sum() / mask.sum(), elementwise_binary_ce_loss.squeeze()


def distortion_loss(pos_logits, neg_logits, mask, device="cpu"):
    logits = cat((pos_logits, neg_logits), dim=-1)
    targets_shape = logits.shape
    logits = logits.reshape(-1, targets_shape[-1])
    targets = 1/targets_shape[-1] * ones(logits.shape, device=device)
    elementwise_distortion_loss = ce_loss(logits, targets).reshape(targets_shape[0], targets_shape[1]) * mask
    return sum(elementwise_distortion_loss) / sum(mask), elementwise_distortion_loss

def click_ssm_order_be_loss(pos_logits, neg_logits, order_labels, mask, device="cpu"):
    ssm_click_loss, ssm_click_elementwise_loss = sampled_softmax_loss(pos_logits, neg_logits, mask, device)
    bce_order_loss, bce_elementwise_loss = binary_cross_entropy_loss(pos_logits, order_labels, mask, device)
    return ssm_click_loss, ssm_click_elementwise_loss, bce_order_loss, bce_elementwise_loss

def click_ssm_distortion_loss(pos_logits, neg_logits, _, mask, device="cpu"):
    ssm_click_loss, ssm_click_elementwise_loss = sampled_softmax_loss(pos_logits, neg_logits, mask, device)
    dis_loss, dis_elementwise_loss = distortion_loss(pos_logits, neg_logits, mask, device)
    return ssm_click_loss, ssm_click_elementwise_loss, dis_loss, dis_elementwise_loss


def weighted_loss(click_loss, second_loss, beta):
    loss_vec = torch.stack([click_loss, second_loss])
    return (loss_vec * beta).sum()


def cosine_loss(click_elementwise_loss, second_elementwise_loss, mask, beta, device="cpu"):
    loss_vectors_for_cosine = torch.stack([click_elementwise_loss, second_elementwise_loss], dim=-1)
    cos_loss = -(cos_sim(beta, loss_vectors_for_cosine) * mask).sum() / mask.sum()
    return cos_loss


def approx_epo_loss(click_elementwise_loss, second_elementwise_loss, mask, beta, device="cpu"):
    elementwise_losses = torch.stack([click_elementwise_loss, second_elementwise_loss], dim=-1)
    weighted_loss = elementwise_losses * beta
    normalized = (weighted_loss / clip(weighted_loss.sum(-1), 1e-6).unsqueeze(-1)) * mask.unsqueeze(-1)
    res_normalized = normalized.reshape(-1, normalized.shape[2])
    cross_entropy = ce_loss(res_normalized, 1 / elementwise_losses.shape[-1] * ones(res_normalized.shape, device=device))
    res = (cross_entropy.reshape(normalized.shape[0], normalized.shape[1], 1) * mask.unsqueeze(-1)).sum() / mask.sum()
    return res


def calc_loss(
    loss_fn,
    x_hat,
    click_labels,
    order_labels,
    uniform_negatives,
    in_batch_negatives,
    mask,
    embeddings,
    sampling_style,
    final_activation,
    topk_sampling=False,
    topk_sampling_k=1000,
    device="cpu",
):
    pos_logits, neg_logits = lookup_and_multiply(
        x_hat,
        click_labels,
        uniform_negatives,
        in_batch_negatives,
        embeddings,
        sampling_style,
    )
    if topk_sampling:
        neg_logits, _ = torch.topk(neg_logits, k=topk_sampling_k, dim=-1)
    pos_scores, neg_scores = final_activation(pos_logits), final_activation(neg_logits)
    return loss_fn(pos_scores, neg_scores, order_labels, mask, device)
