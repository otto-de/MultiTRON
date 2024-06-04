import json
from argparse import ArgumentParser

import mlflow

from src.train import train_multitron

def read_stats(data_dir, dataset):
    with open(f"{data_dir}/{dataset}/{dataset}_stats.json", "r") as f:
        stats = json.load(f)
        train_stats = stats["train"]
        test_stats = stats["test"]
    return train_stats, test_stats, stats["num_items"]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config-filename", type=str)
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--data-dir", type=str, default="datasets")
    args = parser.parse_args()

    with open(f"{args.config_dir}/{args.config_filename}.json", "r") as f:
        config = json.load(f)

    train_stats, test_stats, num_items = read_stats(args.data_dir, config["dataset"])


    trainer, model, train_loader, test_loader = train_multitron(config, args.data_dir, train_stats, test_stats, num_items)


    if config["overfit_batches"] > 0:
        test_loader = train_loader

    mlflow.pytorch.autolog(log_every_n_epoch=1
                           , log_every_n_step=10)

    with mlflow.start_run(run_name=args.config_filename) as run:
        mlflow.log_params(config)
        trainer.fit(model, train_loader, test_loader)
