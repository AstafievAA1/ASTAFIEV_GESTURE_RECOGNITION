import torch
import torch.nn as nn
class GestureLSTM(nn.Module):
    def __init__(
            self,
            input_size: int = 152,
            hidden_size: int = 128,
            num_layers: int = 3,
            num_classes: int = 8,
            dropout: float = 0.5,
    ):
        super(GestureLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.LayerNorm(input_size),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_size = hidden_size

        self.fc1 = nn.Linear(lstm_out_size, 128)
        self.ln1 = nn.LayerNorm(128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(64, num_classes)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

        for module in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def _fc_layers(self, h: torch.Tensor) -> torch.Tensor:
        out = self.dropout1(self.relu1(self.ln1(self.fc1(h))))
        out = self.dropout2(self.relu2(self.ln2(self.fc2(out))))
        return self.fc3(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        h_last = lstm_out[:, -1, :]
        return self._fc_layers(h_last)

    def forward_stateful(self, h_last: torch.Tensor) -> torch.Tensor:
        return self._fc_layers(h_last)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)




