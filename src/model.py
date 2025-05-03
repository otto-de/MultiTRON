import numpy as np
import pytorch_lightning as pl
import torch
from torch import concat, diag, logical_and, logical_or, nn, tensor, tile
from torch.nn import Dropout, Linear

from src.evaluate import validate_batch_per_timestamp
from src.loss import (approx_epo_loss, calc_loss, click_ssm_order_be_loss, click_ssm_distortion_loss, weighted_loss)


class DynamicPositionEmbedding(torch.nn.Module):

    def __init__(self, max_len, dimension):
        super(DynamicPositionEmbedding, self).__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(max_len, dimension)
        self.pos_indices = torch.arange(0, self.max_len, dtype=torch.int)
        self.register_buffer("pos_indices_const", self.pos_indices)

    def forward(self, x, device="cpu"):
        seq_len = x.shape[1]
        return self.embedding(self.pos_indices_const[-seq_len:]) + x


class MultiTron(pl.LightningModule):

    def __init__(
        self,
        hidden_size,
        dropout_rate,
        max_len,
        num_items,
        batch_size,
        sampling_style,
        topk_sampling=False,
        topk_sampling_k=1000,
        learning_rate=0.001,
        num_layers=2,
        loss="ssm_and_be",
        optimizer="adam",
        output_bias=False,
        share_embeddings=True,
        final_activation=False,
        beta=[0.5, 0.5],
        regularization_penalty=0.2,
    ):
        super(MultiTron, self).__init__()
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_items = num_items
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.max_len = max_len
        self.output_bias = output_bias
        self.share_embeddings = share_embeddings
        self.beta = beta

        self.future_mask = torch.triu(torch.ones(self.max_len, self.max_len) * float("-inf"), diagonal=1)
        self.fixed_beta = None
        self.register_buffer("future_mask_const", self.future_mask)
        self.register_buffer("seq_diag_const", ~diag(torch.ones(self.max_len, dtype=torch.bool)))
        self.register_buffer("bias_ones", torch.ones([self.batch_size, self.max_len, 1]))
        self.register_buffer("regularization_penalty", torch.tensor(regularization_penalty))
        if output_bias and share_embeddings:
            self.item_embedding = nn.Embedding(num_items + 1, hidden_size + 1, padding_idx=0)
        else:
            self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.positional_embedding_layer = DynamicPositionEmbedding(self.max_len, hidden_size)

        torch.nn.init.xavier_uniform_(self.item_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.positional_embedding_layer.embedding.weight.data)

        if share_embeddings:
            self.output_embedding = self.item_embedding
        elif (not share_embeddings) and output_bias:
            self.output_embedding = nn.Embedding(num_items + 1, hidden_size + 1, padding_idx=0)
        else:
            self.output_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)

        self.norm = nn.LayerNorm([hidden_size])
        self.input_dropout = Dropout(dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=1,
            dim_feedforward=hidden_size,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers, norm=self.norm)

        self.merge_attn_mask = True
        if final_activation:
            self.final_activation = nn.ELU(0.5)
        else:
            self.final_activation = nn.Identity()

        self.loss_fn = loss
        if self.loss_fn == "ssm_and_be":
            self.loss = click_ssm_order_be_loss
            self.second_loss_prefix = "order"
        elif self.loss_fn == "ssm_and_distortion":
            self.loss = click_ssm_distortion_loss
            self.second_loss_prefix = "distortion"
        else:
            raise ValueError("MultiTron supports ssm_and_be and ssm_and_distortion as loss functions")

        self.sampling_style = sampling_style
        self.topk_sampling = topk_sampling
        self.topk_sampling_k = topk_sampling_k
        self.optimizer = optimizer

        self.linear = Linear(self.hidden_size + len(self.beta), hidden_size)
        self.save_hyperparameters()

    def merge_attn_masks(self, padding_mask):
        batch_size = padding_mask.shape[0]
        seq_len = padding_mask.shape[1]

        if not self.merge_attn_mask:
            return self.future_mask_const[:seq_len, :seq_len]

        padding_mask_broadcast = ~padding_mask.bool().unsqueeze(1)
        future_masks = tile(self.future_mask_const[:seq_len, :seq_len], (batch_size, 1, 1))
        merged_masks = logical_or(padding_mask_broadcast, future_masks)
        # Always allow self-attention to prevent NaN loss
        # See: https://github.com/pytorch/pytorch/issues/41508
        diag_masks = tile(self.seq_diag_const[:seq_len, :seq_len], (batch_size, 1, 1))
        return logical_and(diag_masks, merged_masks)

    def forward(self, item_indices, beta, mask):
        att_mask = self.merge_attn_masks(mask)
        items = (self.item_embedding(item_indices)[:, :, :-1]
                 if self.output_bias and self.share_embeddings else self.item_embedding(item_indices))

        beta_for_each_time_step = beta.repeat([self.max_len * item_indices.shape[0],
                                               1]).reshape([item_indices.shape[0], item_indices.shape[1], 2])

        items = self.linear(torch.concat([items, beta_for_each_time_step], dim=-1))

        x = items * np.sqrt(self.hidden_size / 2)

        x = self.positional_embedding_layer(x)
        x = self.encoder(self.input_dropout(x), att_mask)
        # [batchsize, seqlen, 200], [1 oder batch_size, 200, 200]

        return concat([x, self.bias_ones], dim=-1) if self.output_bias else x

    def training_step(self, batch, _):
        beta = torch.distributions.dirichlet.Dirichlet(tensor([self.beta], device=self.device)).sample()

        x_hat = self.forward(batch["clicks"], beta, batch["mask"])
        click_loss, click_elemetwise_loss, second_loss, second_elementwise_loss = calc_loss(
            self.loss,
            x_hat,
            batch["click_labels"],
            batch["order_labels"],
            batch["uniform_negatives"],
            batch["in_batch_negatives"],
            batch["mask"],
            self.output_embedding,
            self.sampling_style,
            self.final_activation,
            self.topk_sampling,
            self.topk_sampling_k,
            self.device,
        )

        non_uniformity_loss = approx_epo_loss(click_elemetwise_loss, second_elementwise_loss, batch["mask"], beta, self.device)
        train_loss = weighted_loss(click_loss, second_loss, beta) + self.regularization_penalty * non_uniformity_loss

        self.log("train_loss", train_loss)
        self.log("train_click_loss", click_loss)
        self.log(f"train_{self.second_loss_prefix}_loss", second_loss)
        self.log("train_non_uniformity_loss", non_uniformity_loss)
        return train_loss

    def validation_step(self, batch, _batch_idx):
        beta = torch.distributions.dirichlet.Dirichlet(tensor([self.beta], device=self.device)).sample()
        x_hat = self.forward(batch["clicks"], beta, batch["mask"])
        click_loss, click_elemetwise_loss, second_loss, second_elementwise_loss = calc_loss(
            self.loss,
            x_hat,
            batch["click_labels"],
            batch["order_labels"],
            batch["uniform_negatives"],
            batch["in_batch_negatives"],
            batch["mask"],
            self.output_embedding,
            self.sampling_style,
            self.final_activation,
            self.topk_sampling,
            self.topk_sampling_k,
            self.device,
        )

        non_uniformity_loss = approx_epo_loss(click_elemetwise_loss, second_elementwise_loss, batch["mask"], beta, self.device)
        test_loss = weighted_loss(click_loss, second_loss, beta) + self.regularization_penalty * non_uniformity_loss

        cut_offs = tensor([20], device=self.device)

        click_recall, order_recall, order_density = (validate_batch_per_timestamp(batch, x_hat, self.output_embedding, cut_offs))

        self.log("val_loss", test_loss)
        self.log("val_click_loss", click_loss)
        self.log(f"val_{self.second_loss_prefix}_loss", second_loss)
        self.log("val_seq_len", x_hat.shape[1])
        self.log("val_click_recall", click_recall)
        self.log("val_order_recall", order_recall)
        self.log("val_order_density", order_density)
        self.log("val_product_recall_od", click_recall*order_density)
        self.log("val_non_uniformity_loss", non_uniformity_loss)

    def test_step(self, batch, batch_idx):
        beta = torch.tensor([self.fixed_beta], device=self.device)

        x_hat = self.forward(batch["clicks"], beta, batch["mask"])
        click_loss, click_elemetwise_loss, second_loss, second_elementwise_loss = calc_loss(
            self.loss,
            x_hat,
            batch["click_labels"],
            batch["order_labels"],
            batch["uniform_negatives"],
            batch["in_batch_negatives"],
            batch["mask"],
            self.output_embedding,
            self.sampling_style,
            self.final_activation,
            self.topk_sampling,
            self.topk_sampling_k,
            self.device,
        )

        non_uniformity_loss = approx_epo_loss(click_elemetwise_loss, second_elementwise_loss, batch["mask"], beta, self.device)
        test_loss = weighted_loss(click_loss, second_loss, beta) + self.regularization_penalty * non_uniformity_loss

        cut_offs = tensor([20], device=self.device)

        click_recall, order_recall, order_density = (validate_batch_per_timestamp(batch, x_hat, self.output_embedding, cut_offs))

        self.log("test_loss", test_loss)
        self.log("test_click_loss", click_loss)
        self.log(f"test_{self.second_loss_prefix}_loss", second_loss)
        self.log("test_seq_len", x_hat.shape[1])
        self.log("test_click_recall", click_recall)
        self.log("test_order_recall", order_recall)
        self.log("test_order_density", order_density)
        self.log("test_product_recall_od", click_recall*order_density)
        self.log("test_non_uniformity_loss", non_uniformity_loss)

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "adagrad":
            optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Optimizer not supported, please use sgd, adam or adagrad")
        return optimizer
