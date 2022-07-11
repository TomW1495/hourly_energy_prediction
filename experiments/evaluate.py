
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from argparse import ArgumentParser


from src.config import DATA_PATH
from src.models.lstm import LSTMRegressor
from energy_dataset.data import BaseDataset
from src.parameters import params


def evaluate(model, test_x, test_y, label_scalers, device):
    model.eval()
    outputs = []
    targets = []
    sMAPE = 0
    j = 1
    for i in test_x.keys():
        print(f"Evaluating File {j} of {len(test_x)}", end="\r")
        input = torch.from_numpy(np.array(test_x[i])).float()
        label = torch.from_numpy(np.array(test_y[i])).float()
        with torch.no_grad():
            y_hat = model(input)
        outputs.append(label_scalers[i].inverse_transform(y_hat.cpu().detach().numpy()).reshape(-1))
        targets.append(label_scalers[i].inverse_transform(label.numpy()).reshape(-1))
        j += 1

    sMAPE = 0
    for i in range(len(outputs)):
        sMAPE += np.mean(abs(outputs[i]-targets[i])/(targets[i]+outputs[i])/2)/len(outputs)
    print("sMAPE: {}%".format(sMAPE * 100))
    return outputs, targets, sMAPE

def main():
    arguments_parser = ArgumentParser()
    arguments_parser.add_argument("--model_path", required=True,
                                  help="Model Folder Path")
    args = arguments_parser.parse_args()

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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

    model = LSTMRegressor.load_from_checkpoint(args.model_path,
                                               n_features = params['n_features'],
                                               hidden_size = params['hidden_size'],
                                               lookback = params['lookback'],
                                               batch_size = params['batch_size'],
                                               criterion = params['criterion'],
                                               num_layers = params['num_layers'],
                                               dropout = params['dropout'],
                                               learning_rate = params['learning_rate'])
    model = model
    base_dataset = BaseDataset(DATA_PATH, lookback = params['lookback'])
    outputs, targets, sMAPE = evaluate(model,
                                       base_dataset.test_scaler_x,
                                       base_dataset.test_scaler_y,
                                       base_dataset.label_scalers,
                                       device)


    plt.plot(outputs[6][:100], "-o", color="g", label="Predicted")
    plt.plot(targets[6][:100], color="b", label="Actual")
    plt.ylabel('Energy Consumption (MW)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()