import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import pandas as pd
from pprint import pprint
import ssl

from data import CellDataset
from models import CellModel

ssl._create_default_https_context = ssl._create_unverified_context


def main(base_dir, model_name, model_encoder, batch_size, load_path="", load=False):
    # init train, val, test sets
    data_df = pd.read_csv(os.path.join(base_dir, "data.csv"))
    train_dataset = CellDataset(data_df, "train")
    valid_dataset = CellDataset(data_df, "val")
    test_dataset = CellDataset(data_df, "test")

    # It is a good practice to check datasets don`t intersects with each other
    assert set(test_dataset.filenames).isdisjoint(set(train_dataset.filenames))
    assert set(test_dataset.filenames).isdisjoint(set(valid_dataset.filenames))
    assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames))

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    n_cpu = os.cpu_count()
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpu
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu
    )

    # Model
    model = CellModel(model_name, model_encoder, 1, 1)

    if load:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["state_dict"])
        print("Model loaded successfully")

    checkpoint_callback = ModelCheckpoint(
        dirpath=base_dir,
        monitor="valid_dataset_iou",
        filename=model_name
        + "-"
        + model_encoder
        + "-epoch{epoch:03d}-{valid_dataset_iou:.4f}",
        save_top_k=2,
        mode="max",
        auto_insert_metric_name=False,
    )

    # Training
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=100,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )

    # Validation and test metrics
    # run validation dataset
    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    pprint(valid_metrics)

    # run test dataset
    test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
    pprint(test_metrics)


if __name__ == "__main__":
    base_dir = os.path.join("data", "Fluo-N3DH-SIM+_splitted")
    # base_dir = "/content/drive/MyDrive/Colab Notebooks/3D segmentation/Fluo-N3DH-SIM+_splitted_filtered"
    model_name = "Unet"
    model_encoder = "resnet18"
    batch_size = 8
    load = False
    load_path = ""
    main(base_dir, model_name, model_encoder, batch_size, load_path, load)
