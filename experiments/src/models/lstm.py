# Neural Networks
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torchmetrics import SymmetricMeanAbsolutePercentageError
from ..constants import Metrics

class LSTMRegressor(LightningModule):
    '''
    Standard PyTorch Lightning module:
    https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    '''
    def __init__(self,
                 n_features,
                 hidden_size,
                 lookback,
                 batch_size,
                 num_layers,
                 dropout,
                 learning_rate,
                 criterion):
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.lookback = lookback
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate

        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:,-1])
        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("training_loss", loss)
        self.log('avg_training_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return {Metrics.VAL_LOSS: loss, Metrics.PREDICTIONS: y_hat, Metrics.TARGETS: y}

    def validation_epoch_end(self, outs):
        avg_val_loss = torch.stack([o[Metrics.VAL_LOSS] for o in outs]).mean()
        predictions = torch.stack([o[Metrics.PREDICTIONS] for o in outs])
        targets = torch.stack([o[Metrics.TARGETS] for o in outs])
        self.log("avg_val_loss", avg_val_loss)
        smape = SymmetricMeanAbsolutePercentageError()
        smape_metric = smape(predictions, targets)
        print(f"Val SMAPE: {smape_metric}, Val Loss: {avg_val_loss}\n")
        self.log("Val SMAPE", avg_val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return {Metrics.TEST_LOSS: loss, Metrics.PREDICTIONS: y_hat, Metrics.TARGETS: y}

    def test_epoch_end(self, outs):
        avg_test_loss = torch.stack([o[Metrics.TEST_LOSS] for o in outs]).mean()
        predictions = torch.stack([o[Metrics.PREDICTIONS] for o in outs])
        targets = torch.stack([o[Metrics.TARGETS] for o in outs])
        self.log("avg_test_loss", avg_test_loss)
        smape = SymmetricMeanAbsolutePercentageError()
        smape_metric = smape(predictions, targets)
        print(f"Test SMAPE: {smape_metric}, Test Loss: {avg_test_loss}\n")
        self.log("Test SMAPE", smape_metric)
