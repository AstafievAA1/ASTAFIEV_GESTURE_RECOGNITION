import torch
import numpy as np
from typing import Tuple, Optional

class StatefulLSTMInference:
    def __init__(self, trained_model, device='cpu'):
        self.model = trained_model
        self.device = device
        self.model.eval()
        self.num_layers = trained_model.num_layers
        self.hidden_size = trained_model.hidden_size
        self.hidden_state: Optional[torch.Tensor] = None
        self.cell_state: Optional[torch.Tensor] = None
        print(" Stateful LSTM Inference инициализирован")
        print(f"  Layers:      {self.num_layers}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Device:      {self.device}")

    def reset_state(self):
        self.hidden_state = None
        self.cell_state = None

    def _init_state(self, batch_size: int = 1):
        self.hidden_state = torch.zeros(
            self.num_layers, batch_size, self.hidden_size
        ).to(self.device)
        self.cell_state = torch.zeros(
            self.num_layers, batch_size, self.hidden_size
        ).to(self.device)

    def predict_single_frame(self, keypoints: np.ndarray) -> Tuple[int, np.ndarray]:
        frame_tensor = torch.FloatTensor(keypoints).unsqueeze(0).unsqueeze(0).to(self.device)
        if self.hidden_state is None:
            self._init_state(batch_size=1)
        with torch.no_grad():
            frame_projected = self.model.input_proj(frame_tensor)
            lstm_out, (h_new, c_new) = self.model.lstm(
                frame_projected,
                (self.hidden_state, self.cell_state)
            )
            self.hidden_state = h_new.detach()
            self.cell_state = c_new.detach()
            h_last = h_new[-1, :, :]
            out = self.model.forward_stateful(h_last)
            probabilities = torch.softmax(out, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            probs = probabilities[0].cpu().numpy()
        return predicted_class, probs