import numpy as np

class NoddingDetector:
    def __init__(self, nod_threshold=15, moving_avg_window=10, tilt_threshold=10, side_drop_threshold=10):
        self.nod_threshold = nod_threshold
        self.moving_avg_window = moving_avg_window
        self.tilt_threshold = tilt_threshold
        self.side_drop_threshold = side_drop_threshold

        self.y_positions = []
        self.eye_tilt_history = []

    def detect(self, landmarks):
        nose_y = landmarks[33][1]
        self.y_positions.append(nose_y)
        if len(self.y_positions) > self.moving_avg_window:
            self.y_positions.pop(0)

        if len(self.y_positions) < self.moving_avg_window:
            return "Normal"

        y_movement = self.y_positions[-1] - self.y_positions[0]

        # 좌우 기울임 (양 눈 높이 차이)
        eye_left_y = landmarks[36][1]
        eye_right_y = landmarks[45][1]
        eye_tilt = eye_left_y - eye_right_y  # 음수 = 왼쪽으로 기울임, 양수 = 오른쪽

        self.eye_tilt_history.append(eye_tilt)
        if len(self.eye_tilt_history) > self.moving_avg_window:
            self.eye_tilt_history.pop(0)

        tilt_change = self.eye_tilt_history[-1] - self.eye_tilt_history[0]

        # 조건 1: 아래로 숙이는 끄덕임
        if y_movement > self.nod_threshold:
            return "Nodding"

        # 조건 2: 고개가 서서히 한쪽으로 기울어진 경우
        elif abs(tilt_change) > self.side_drop_threshold:
            return "Nodding_Side"

        else:
            return "Normal"

