import json

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from src.sample import (sample_in_batch_negatives, sample_uniform, sample_uniform_negatives_with_shape)
from src.utils import get_offsets


def label_session(session, interactions_to_work_on={"clicks", "orders"}, max_seqlen=50):
    session = [event for event in session if event["type"] in interactions_to_work_on]

    future_orders = set()
    clicks = []
    order_labels = []
    for event in session[::-1]:
        if event["type"] == "orders":
            future_orders.add(event["aid"])

        elif event["type"] == "clicks":

            if event["aid"] in future_orders:
                order_labels = [1.0] + order_labels
            else:
                order_labels = [0.0] + order_labels

            clicks = [event["aid"]] + clicks

    clicks = clicks[-(max_seqlen + 1):]

    order_labels = order_labels[-(max_seqlen + 1):]

    click_labels = clicks[1:]
    clicks = clicks[:-1]
    order_labels = order_labels[1:]
    mask = len(clicks) * [1.0]

    session_len = min(len(clicks), max_seqlen)

    return clicks, click_labels, order_labels, session_len, mask


class MultiTronDataset(Dataset):

    def __init__(
        self,
        sessions_path,
        total_sessions,
        num_items,
        max_seqlen,
        num_uniform_negatives=1,
        num_in_batch_negatives=0,
        reject_uniform_session_items=False,
        reject_in_batch_items=True,
        sampling_style="eventwise",
        shuffling_style="no_shuffling",
        interactions_to_work_on={"clicks", "orders"},
        repeat_data=1,
    ):
        self.session_path = sessions_path
        self.total_sessions = total_sessions * repeat_data
        self.num_items = num_items
        self.max_seqlen = max_seqlen
        self.shuffling_style = shuffling_style
        self.num_uniform_negatives = num_uniform_negatives
        self.num_in_batch_negatives = num_in_batch_negatives
        self.reject_uniform_session_items = reject_uniform_session_items
        self.reject_in_batch_items = reject_in_batch_items
        self.sampling_style = sampling_style
        self.line_offsets = (get_offsets(sessions_path)[:self.total_sessions] * repeat_data)
        self.interactions_to_work_on = interactions_to_work_on
        self.repeat_data = repeat_data
        assert self.sampling_style in {"eventwise", "sessionwise", "batchwise"}
        assert (len(self.line_offsets) == self.total_sessions), f"{len(self.line_offsets)} != {self.total_sessions}"

    def __len__(self):
        return self.total_sessions

    def __getitem__(self, idx):
        with open(self.session_path, "rt") as f:

            if self.shuffling_style == "shuffle_with_replacement":
                idx = np.random.randint(0, self.total_sessions)

            f.seek(self.line_offsets[idx])
            line = f.readline()
            session = json.loads(line)
            session = session["events"]

            assert sorted(session, key=lambda d: d["ts"]) == session

            clicks, click_labels, order_labels, session_len, mask = label_session(session, self.interactions_to_work_on,
                                                                                  self.max_seqlen)

            negatives = sample_uniform_negatives_with_shape(
                clicks,
                self.num_items,
                session_len,
                self.num_uniform_negatives,
                self.sampling_style,
                self.reject_uniform_session_items,
            )

            return {
                "clicks": clicks,
                "click_labels": click_labels,
                "order_labels": order_labels,
                "session_len": session_len,
                "uniform_negatives": negatives.tolist(),
                "mask": mask,
            }

    def dynamic_collate(self, batch):
        batch_clicks = list()
        batch_mask = list()
        batch_click_labels = list()
        batch_order_labels = list()
        batch_session_len = list()
        batch_positives = list()
        max_len = self.max_seqlen
        batch_uniform_negatives = list()
        in_batch_negatives = list()

        for item in batch:
            session_len = item["session_len"]

            batch_clicks.append((max_len - session_len) * [0] + item["clicks"])

            batch_click_labels.append((max_len - session_len) * [0] + item["click_labels"])
            batch_order_labels.append((max_len - session_len) * [0] + item["order_labels"])

            batch_mask.append((max_len - session_len) * [0.0] + item["mask"])
            batch_session_len.append(session_len)
            batch_positives.extend(item["clicks"])

            if self.sampling_style == "eventwise":
                batch_uniform_negatives.append((max_len - session_len) * [[0] * self.num_uniform_negatives] +
                                               item["uniform_negatives"])
            elif self.sampling_style == "sessionwise":
                batch_uniform_negatives.append(item["uniform_negatives"])

        if self.sampling_style == "batchwise":
            batch_uniform_negatives = sample_uniform(
                self.num_items,
                [self.num_uniform_negatives],
                set(batch_positives),
                self.reject_in_batch_items,
            )

        in_batch_negatives = sample_in_batch_negatives(
            batch_positives,
            self.num_in_batch_negatives,
            batch_session_len,
            self.reject_in_batch_items,
        )

        return {
            "clicks": torch.tensor(batch_clicks, dtype=torch.long),
            "click_labels": torch.tensor(batch_click_labels, dtype=torch.long),
            "order_labels": torch.tensor(batch_order_labels, dtype=torch.float),
            "mask": torch.tensor(batch_mask, dtype=torch.float),
            "session_len": torch.tensor(batch_session_len, dtype=torch.long),
            "in_batch_negatives": torch.tensor(in_batch_negatives, dtype=torch.long),
            "uniform_negatives": torch.tensor(batch_uniform_negatives, dtype=torch.long),
        }
