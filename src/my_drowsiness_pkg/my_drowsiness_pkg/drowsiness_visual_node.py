# drowsiness_visual_node.py (action vector 제거, 모델 구조 직접 포함)

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import cv2
import torch
from torchvision import transforms
import numpy as np
import os

class CNNEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, 1, 1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, 1, 1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, 1, 1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.cnn(x)
        return x.view(x.size(0), -1)

class DrowsinessCNNLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        self.lstm = torch.nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.fc = torch.nn.Linear(135, 2)  # action vector 제거됨
    # forward 함수 수정
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.encoder(x)
        feats = feats.view(B, T, -1)
        out, _ = self.lstm(feats)
        out = out[:, -1, :]

        # 임시 action vector (0벡터로 대체, 길이 7)
        dummy_action = torch.zeros((B, 7)).to(out.device)

        out = torch.cat([out, dummy_action], dim=1)
        return self.fc(out)

class DrowsinessVisualNode(Node):
    def __init__(self):
        super().__init__('drowsiness_visual_node')
        self.publisher = self.create_publisher(Int32, '/drowsiness/state', 10)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


        self.model = DrowsinessCNNLSTM()
        model_path = os.path.join(os.path.dirname(__file__), 'weights/cnn_lstm_drowsiness_1.pth')
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

        self.T = 5
        self.buffer = []

        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # ✅ PIL 변환 필수
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])


        self.timer = self.create_timer(0.033, self.timer_callback)
        self.get_logger().info('🟢 DrowsinessVisualNode started!')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Camera read failed.')
            return

        self.buffer.append(frame)
        if len(self.buffer) > self.T:
            self.buffer.pop(0)

        state_label = "Waiting..."

        if len(self.buffer) == self.T:
            imgs = [self.transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in self.buffer]
            seq = torch.stack(imgs).unsqueeze(0)
            with torch.no_grad():
                output = self.model(seq)
                pred = torch.argmax(output, dim=1).item()

            state_label = "Drowsy" if pred == 1 else "Awake"
            self.publisher.publish(Int32(data=pred))

        display = frame.copy()
        color = (0, 0, 255) if state_label == "Drowsy" else (0, 255, 0)
        cv2.putText(display, f"State: {state_label}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        if state_label == "Drowsy":
            cv2.putText(display, "\u26a0\ufe0f DROWSINESS DETECTED", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow("Drowsiness Detection", display)
        cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DrowsinessVisualNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
