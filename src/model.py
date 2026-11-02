import torch.nn as nn


class LSTMRegressor(nn.Module):
    """Same architecture as the course project (2-layer LSTM, hidden 64)."""

    def __init__(self, input_size=24, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
