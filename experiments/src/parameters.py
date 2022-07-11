import torch.nn as nn

params = dict(
    lookback = 90,
    batch_size = 512,
    criterion = nn.MSELoss(),
    max_epochs = 50,
    n_features = 5,
    num_workers = 8,
    hidden_size = 100,
    num_layers = 2,
    dropout = 0.2,
    learning_rate = 0.001,
)