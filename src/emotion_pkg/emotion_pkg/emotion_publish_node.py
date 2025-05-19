import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import mediapipe as mp
import torch.nn as nn
import os

# ✅ 감정 레이블 정의 (3클래스)
EMOTION_LABELS = {
    0: "positive",
    1: "neutral",
    2: "negative"
}

# ✅ 3클래스 DeepCNN 모델 정의
class DeepCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(DeepCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ✅ ROS 노드 클래스
class EmotionPublisherNode(Node):
    def __init__(self):
        super().__init__('emotion_publisher_node')
        self.publisher_ = self.create_publisher(String, '/emotion/status', 10)

        # 모델 준비
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepCNN()
        model_path = "/home/sienna/Workspace/dream/src/emotion_pkg/emotion_pkg/model/korean_emotion_model_3class.pth"  # 모델 경로 수정
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        # 전처리 정의 (RGB)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Mediapipe 얼굴 검출
        self.face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

        # 웹캠 열기
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("웹캠을 열 수 없습니다.")
            exit()

        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("프레임을 읽을 수 없습니다.")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_detection.process(frame_rgb)

        output_frame = frame.copy()

        if result.detections:
            for detection in result.detections:
                h, w, _ = frame.shape
                bbox = detection.location_data.relative_bounding_box
                x_min = int(bbox.xmin * w)
                y_min = int(bbox.ymin * h)
                box_width = int(bbox.width * w)
                box_height = int(bbox.height * h)

                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_min + box_width)
                y_max = min(h, y_min + box_height)

                face_img = frame[y_min:y_max, x_min:x_max]
                if face_img.size == 0:
                    continue

                input_tensor = self.transform(face_img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.model(input_tensor)
                    probs = F.softmax(output, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    emotion = EMOTION_LABELS[pred]
                    confidence = probs[0][pred].item()

                # ✅ 토픽 발행
                msg = String()
                msg.data = f"{emotion} ({confidence*100:.1f}%)"
                self.publisher_.publish(msg)
                self.get_logger().info(f"발행: {msg.data}")

                # ✅ 시각화
                cv2.rectangle(output_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label_text = f"{emotion} ({confidence*100:.1f}%)"
                cv2.putText(output_frame, label_text, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                break

        cv2.imshow("Emotion Recognition (3-Class)", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    def destroy_node(self):
        super().destroy_node()
        self.cap.release()
        self.face_detection.close()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = EmotionPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
