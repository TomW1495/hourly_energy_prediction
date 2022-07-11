import argparse
import sys
import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.energy_data_module import EnergyDataModule
from src.models.lstm import LSTMRegressor
from src.parameters import params


def main():
    seed_everything(1)
    logger = TensorBoardLogger("tb_logs", name="LSTM Energy Prediction")

    trainer = Trainer(
        callbacks=[EarlyStopping(monitor="avg_val_loss", mode="min", patience=10)],
        #default_root_dir="checkpoints/",
        max_epochs=params['max_epochs'],
        logger=logger,
        gpus=torch.cuda.device_count(),
        log_every_n_steps=1,
        progress_bar_refresh_rate=2,
    )

    model = LSTMRegressor(
        n_features = params['n_features'],
        hidden_size = params['hidden_size'],
        lookback = params['lookback'],
        batch_size = params['batch_size'],
        criterion = params['criterion'],
        num_layers = params['num_layers'],
        dropout = params['dropout'],
        learning_rate = params['learning_rate']
    )

    dm = EnergyDataModule(
        lookback = params['lookback'],
        batch_size = params['batch_size'],
        num_workers = params['num_workers']
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()