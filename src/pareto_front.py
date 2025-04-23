import json
import os
from argparse import ArgumentParser

import torch
from pytorch_lightning.trainer.trainer import Trainer
from scipy.stats import beta
from torch.utils.data import DataLoader

from src.dataset import MultiTronDataset
from src.model import MultiTron


def read_stats(data_dir, dataset):
    with open(f"{data_dir}/{dataset}/{dataset}_stats.json", "r") as f:
        stats = json.load(f)
        train_stats = stats["train"]
        test_stats = stats["test"]
    return train_stats, test_stats, stats["num_items"]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config-filename", type=str, default="config")
    parser.add_argument("--checkpoint", type=str, default="model.ckpt")
    parser.add_argument("--data-dir", type=str, default="datasets")
    parser.add_argument("--number-points", type=int, default=25)
    args = parser.parse_args()

    with open(f"configs/{args.config_filename}.json", "r") as f:
        config = json.load(f)

    number_points = args.number_points  # number_points + 1 is the total number of points
    dataset = config["dataset"]
    if dataset == "diginetica":
        repeat_data = 16 * 16
        b = beta(config["beta"][0], config["beta"][1])
        fixed_betas = [[float(b.ppf(i / number_points)), float(1 - b.ppf(i / number_points))]
                       for i in range(0, number_points + 1)]
    elif dataset == "yoochoose":
        repeat_data = 16 * 16  # bring down negative sample induced variance in loss because test dataset is small
        b = beta(config["beta"][0], config["beta"][1])
        fixed_betas = [[(i / number_points)**2, 1 - (i / number_points)**2] for i in range(0, number_points + 1)]
    else:
        repeat_data = 1
        b = beta(config["beta"][0], config["beta"][1])
        fixed_betas = [[float(b.ppf(i / number_points)), float(1 - b.ppf(i / number_points))]
                       for i in range(0, number_points + 1)]

    train_stats, test_stats, num_items = read_stats(args.data_dir, dataset)

    test_set = MultiTronDataset(f"{args.data_dir}/{dataset}/{dataset}_test.jsonl",
                                total_sessions=test_stats["num_sessions"],
                                num_items=num_items,
                                max_seqlen=config["max_session_length"],
                                num_in_batch_negatives=config["num_batch_negatives"],
                                num_uniform_negatives=config["num_uniform_negatives"],
                                reject_uniform_session_items=config["reject_uniform_session_items"],
                                reject_in_batch_items=config["reject_in_batch_items"],
                                sampling_style=config["sampling_style"],
                                shuffling_style="no_shuffling",
                                repeat_data=repeat_data)

    test_loader = DataLoader(test_set,
                             drop_last=True,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             pin_memory=True,
                             persistent_workers=True,
                             num_workers=os.cpu_count(),
                             collate_fn=test_set.dynamic_collate)

    trainer = Trainer(config["accelerator"])
    model = MultiTron.load_from_checkpoint(f"{args.checkpoint}", map_location=torch.device("cpu"))
    model.eval()

    tasks = []
    order_weight = []

    for beta in fixed_betas:
        model.fixed_beta = beta
        res = trainer.test(model, test_loader)
        tasks.append(res)
        order_weight.append(beta[1])

    click_loss = [task[0]["test_click_loss"] for task in tasks]
    order_loss = [task[0]["test_order_loss"] for task in tasks]
    non_uniformity_loss = [task[0]["test_non_uniformity_loss"] for task in tasks]
    click_recall = [task[0]["test_click_recall"] for task in tasks]
    order_recall = [task[0]["test_order_recall"] for task in tasks]
    order_density = [task[0]["test_order_density"] for task in tasks]
    product_recall_od = [task[0]["test_product_recall_od"] for task in tasks]

    with open("pareto-front.json", "w") as f:
        f.write(
            json.dumps({
                "order_weight": order_weight,
                "click_loss": click_loss,
                "order_loss": order_loss,
                "non_uniformity_loss": non_uniformity_loss,
                "click_recall": click_recall,
                "order_recall": order_recall,
                "order_density": order_density,
                "product_recall_od": product_recall_od
            }))
