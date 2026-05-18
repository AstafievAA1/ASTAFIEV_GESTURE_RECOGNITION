import time
import numpy as np
from collections import deque
from typing import Optional, Tuple
from stateful_lstm_wrapper import StatefulLSTMInference
class OnlineLSTMInferenceV2:
    def __init__(
        self,
        model,
        device,
        actions:               list,
        min_frames_to_decide:  int   = 6,
        confidence_threshold:  float = 0.80,
        smoothing_window:      int   = 5,
        hard_reset_timeout:    float = 1.5,
    ):
        self.lstm_inf  = StatefulLSTMInference(model, device)
        self.actions   = actions
        self.device    = device

        self.min_frames           = min_frames_to_decide
        self.confidence_threshold = confidence_threshold
        self.smoothing_window     = smoothing_window
        self.hard_reset_timeout   = hard_reset_timeout

        self.no_gesture_idx = (
            actions.index('no_gesture') if 'no_gesture' in actions else -1
        )

        self.frame_count        = 0
        self.predictions_buffer = deque(maxlen=smoothing_window)
        self.last_hand_time     = time.monotonic()

    def process(
        self,
        features:  np.ndarray,
        has_hands: bool,
    ) -> Tuple[Optional[str], float, Optional[np.ndarray]]:
        now = time.monotonic()

        if not has_hands:
            if now - self.last_hand_time > self.hard_reset_timeout:
                self._hard_reset()
            return 'no_gesture', 0.0, None

        self.last_hand_time = now

        predicted_class, probs = self.lstm_inf.predict_single_frame(features)
        self.frame_count += 1
        self.predictions_buffer.append(predicted_class)

        if self.frame_count < self.min_frames:
            return None, float(probs[predicted_class]), probs

        classes, counts = np.unique(list(self.predictions_buffer), return_counts=True)
        smoothed_class  = int(classes[np.argmax(counts)])
        confidence      = float(probs[smoothed_class])
        gesture_name    = self.actions[smoothed_class]

        if smoothed_class == self.no_gesture_idx:
            if confidence >= self.confidence_threshold:
                self._soft_reset()
                return gesture_name, confidence, probs
            return None, confidence, probs

        if confidence < self.confidence_threshold:
            return None, confidence, probs

        self._soft_reset()
        return gesture_name, confidence, probs

    def _soft_reset(self):
        self.frame_count        = 0
        self.frames_since_last  = 0
        self.predictions_buffer.clear()

    def _hard_reset(self):
        self.lstm_inf.reset_state()
        self.frame_count        = 0
        self.predictions_buffer.clear()