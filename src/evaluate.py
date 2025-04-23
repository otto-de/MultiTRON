from torch import cumsum, flip, inf, max, stack, sum, topk, where

from src.logits_computation import multiply_head_with_embedding


def calculate_ranks(logits, labels, cutoffs):
    num_logits = logits.shape[-1]
    k = min(num_logits, max(cutoffs).item())
    _, indices = topk(logits, k=k, dim=-1)
    indices = flip(indices, dims=[-1])
    hits = indices == labels.unsqueeze(dim=-1)
    ranks = sum(cumsum(hits, -1), -1) - 1.0
    ranks[ranks == -1] = float("inf")
    return ranks


def pointwise_mrr(ranks, cutoffs, mask):
    res = where(ranks < cutoffs.unsqueeze(-1).unsqueeze(-1), ranks, float("inf"))
    return (1 / (res + 1)) * mask


def pointwise_recall(ranks, cutoffs, mask):
    res = ranks < cutoffs.unsqueeze(-1).unsqueeze(-1)
    return res.float() * mask


def mean_metric(pointwise_metric, mask):
    hits = sum(pointwise_metric, dim=(2, 1))
    return hits / sum(mask).clamp(0.0000005)


def validate_batch_per_timestamp(batch, x_hat, output_embedding, cut_offs):
    recalls_clicks = []
    recalls_orders = []

    for t in range(x_hat.shape[1]):

        positives = batch["click_labels"][:, t]
        logits = multiply_head_with_embedding(x_hat[:, t], output_embedding.weight)

        logits[:, 0] = -inf  # set score for padding item to -inf
        ranks = calculate_ranks(logits, positives, cut_offs)
        pw_rec_clicks = pointwise_recall(ranks, cut_offs, batch["mask"][:, t])
        pw_rec_orders = pointwise_recall(ranks, cut_offs, batch["order_labels"][:, t])

        recalls_clicks.append(pw_rec_clicks.squeeze(dim=1))
        recalls_orders.append(pw_rec_orders.squeeze(dim=1))

    pw_rec_clicks = stack(recalls_clicks, dim=2)
    pw_rec_orders = stack(recalls_orders, dim=2)

    clicks_recall = mean_metric(pw_rec_clicks, batch["mask"])
    orders_recall = mean_metric(pw_rec_orders, batch["order_labels"])
    order_density = mean_metric(pw_rec_clicks* pw_rec_orders, pw_rec_clicks)

    return clicks_recall, orders_recall, order_density
