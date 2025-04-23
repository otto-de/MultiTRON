import os

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader

from src.dataset import MultiTronDataset
from src.model import MultiTron


def train_multitron(config, data_dir, train_stats, test_stats, num_items):
    checkpoint_callback = ModelCheckpoint(save_top_k=config["max_epochs"],
                                          monitor='val_loss',
                                          mode='min',
                                          filename=f'multitron-{config["dataset"]}-' + '{epoch}-{test_loss:.3f}')

    trainer = Trainer(max_epochs=config["max_epochs"],
                      precision=16,
                      limit_val_batches=config["limit_val_batches"],
                      log_every_n_steps=10,
                      accelerator=config["accelerator"],
                      overfit_batches=config["overfit_batches"],
                      callbacks=[checkpoint_callback],
                      check_val_every_n_epoch=1)

    assert 0 <= config["num_batch_negatives"] < config['batch_size']

    train_set = MultiTronDataset(f'{data_dir}/{config["dataset"]}/{config["dataset"]}_train.jsonl',
                                 train_stats["num_sessions"],
                                 num_items=num_items,
                                 max_seqlen=config["max_session_length"],
                                 num_in_batch_negatives=config["num_batch_negatives"],
                                 num_uniform_negatives=config["num_uniform_negatives"],
                                 reject_uniform_session_items=config["reject_uniform_session_items"],
                                 reject_in_batch_items=config["reject_in_batch_items"],
                                 sampling_style=config["sampling_style"],
                                 shuffling_style=config["shuffling_style"])

    test_set = MultiTronDataset(f'{data_dir}/{config["dataset"]}/{config["dataset"]}_test.jsonl',
                                test_stats["num_sessions"],
                                num_items=num_items,
                                max_seqlen=config["max_session_length"],
                                num_in_batch_negatives=config["num_batch_negatives"],
                                num_uniform_negatives=config["num_uniform_negatives"],
                                reject_uniform_session_items=config["reject_uniform_session_items"],
                                reject_in_batch_items=config["reject_in_batch_items"],
                                sampling_style=config["sampling_style"],
                                shuffling_style="no_shuffling")

    shuffle = True if config["shuffling_style"] == "shuffle_without_replacement" else False
    train_loader = DataLoader(train_set,
                              drop_last=True,
                              batch_size=config["batch_size"],
                              shuffle=shuffle,
                              pin_memory=True,
                              persistent_workers=True,
                              num_workers=os.cpu_count(),
                              collate_fn=train_set.dynamic_collate)

    test_loader = DataLoader(test_set,
                             drop_last=True,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             pin_memory=True,
                             persistent_workers=True,
                             num_workers=os.cpu_count(),
                             collate_fn=test_set.dynamic_collate)

    model = MultiTron(hidden_size=config["hidden_size"],
                      dropout_rate=config["dropout"],
                      max_len=config["max_session_length"],
                      num_items=num_items,
                      batch_size=config["batch_size"],
                      sampling_style=config["sampling_style"],
                      topk_sampling=config.get("topk_sampling", False),
                      topk_sampling_k=config.get("topk_sampling_k", 1000),
                      learning_rate=config["lr"],
                      num_layers=config["num_layers"],
                      loss=config["loss"],
                      optimizer=config["optimizer"],
                      output_bias=config["output_bias"],
                      share_embeddings=config["share_embeddings"],
                      beta=config["beta"],
                      regularization_penalty=config["regularization_penalty"])

    return trainer, model, train_loader, test_loader
