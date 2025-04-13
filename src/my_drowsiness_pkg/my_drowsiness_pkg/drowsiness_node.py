# drowsiness_node.py (수정본)
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
import torch
import torch.nn as nn
import numpy as np
import cv2
from cv_bridge import CvBridge
from torchvision import transforms

# 모델 정의 (기본 CNN + LSTM 구조)
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.cnn(x)
        return x.view(x.size(0), -1)

class DrowsinessCNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128 + 7, 2)  # action vector dummy 포함

    def forward(self, x, action_vec):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.encoder(x)
        feats = feats.view(B, T, -1)
        out, _ = self.lstm(feats)
        out = out[:, -1, :]
        out = torch.cat([out, action_vec], dim=1)
        return self.fc(out)

class DrowsinessNode(Node):
    def __init__(self):
        super().__init__('drowsiness_node')
        self.subscription = self.create_subscription(Image, '/sequence/image_sequence', self.callback, 10)
        self.publisher = self.create_publisher(Int32, '/drowsiness/state', 10)
        self.bridge = CvBridge()
        self.buffer = []
        self.seq_len = 5

        model_path = self.get_model_path()
        self.model = DrowsinessCNNLSTM()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.get_logger().info("🟢 DrowsinessNode initialized!")

    def get_model_path(self):
        import os
        return os.path.join(
            os.getenv('AMENT_PREFIX_PATH').split(':')[0],
            'share/my_drowsiness_pkg/weights/cnn_lstm_drowsiness_1.pth'
        )

    def callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        self.buffer.append(frame)
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

        if len(self.buffer) == self.seq_len:
            with torch.no_grad():
                imgs = [self.transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in self.buffer]
                seq = torch.stack(imgs).unsqueeze(0)  # (1, T, C, H, W)
                dummy_action = torch.zeros((1, 7))  # action vector dummy
                output = self.model(seq, dummy_action)
                pred = torch.argmax(output, dim=1).item()
                self.publisher.publish(Int32(data=pred))
                self.get_logger().info(f"🔎 Predicted: {pred}")

def main(args=None):
    rclpy.init(args=args)
    node = DrowsinessNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
