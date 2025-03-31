# ✅ 실시간 얼굴 시퀀스 기반 CNN + LSTM 졸음 인식 (캘리브레이션 포함)

import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from facenet_pytorch import MTCNN
from collections import deque
import time

# ------------------------------
# 모델 정의 (CNN + LSTM)
# ------------------------------
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )

    def forward(self, x):
        x = self.cnn(x)
        return x.view(x.size(0), -1)

class DrowsinessCNNLSTM(nn.Module):
    def __init__(self, action_dim, num_classes):
        super().__init__()
        self.encoder = CNNEncoder()
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128 + action_dim, num_classes)

    def forward(self, x, action_vec):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.encoder(x)
        feats = feats.view(B, T, -1)
        out, _ = self.lstm(feats)
        out = out[:, -1, :]
        out = torch.cat([out, action_vec], dim=1)
        return self.fc(out)

# ------------------------------
# 설정
# ------------------------------
action_vocab = ["하품", "꾸벅꾸벅졸다", "눈비비기", "눈깜빡이기", "전방주시", "운전하다", "기타"]
action2idx = {a: i for i, a in enumerate(action_vocab)}
action_str = "운전하다"  # 실시간에서는 고정

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# ------------------------------
# 장치 및 모델 초기화
# ------------------------------
mtcnn = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')
face_buffer = deque(maxlen=5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DrowsinessCNNLSTM(action_dim=len(action_vocab), num_classes=2).to(device)
model.load_state_dict(torch.load("cnn_lstm_drowsiness_1.pth", map_location=device))
model.eval()

calibration_weights = torch.ones(2).to(device)  # 초기값: 무보정

# ------------------------------
# RealSense 카메라 시작
# ------------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# ------------------------------
# 캘리브레이션 함수
# ------------------------------
def calibrate(label_name, class_idx):
    print(f"[INFO] '{label_name}' 상태를 5초 동안 유지하세요...")
    buffer = []
    start = time.time()
    while time.time() - start < 5:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        frame = cv2.flip(frame, 1)
        original = frame.copy()

        boxes, _ = mtcnn.detect(frame)
        # 얼굴 탐지 및 크롭
        if boxes is not None:
            x1, y1, x2, y2 = boxes[0].astype(int)
            face = original[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # ✅ 크롭된 얼굴 시각화 (여기서 정의된 이후니까 OK!)
            cv2.imshow("Face Crop", face)

            face_tensor = transform(face)
            face_buffer.append(face_tensor)

            if len(face_buffer) == 5:
                input_seq = torch.stack(list(face_buffer)).unsqueeze(0).to(device)
                action_idx = action2idx.get(action_str, action2idx["기타"])
                action_vec = torch.nn.functional.one_hot(torch.tensor([action_idx]), num_classes=len(action_vocab)).float().to(device)

                with torch.no_grad():
                    output = model(input_seq, action_vec)
                    buffer.append(output.squeeze(0))

        cv2.putText(original, f"Calibrating: {label_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.imshow("Calibrating", original)
        cv2.waitKey(1)

    if buffer:
        avg = torch.stack(buffer, dim=0).mean(dim=0)
        calibration_weights[class_idx] = calibration_weights[class_idx] * (1.5 / (avg[class_idx] + 1e-6))
        print(f"[INFO] {label_name} 캘리브레이션 완료. 보정값: {calibration_weights[class_idx]:.2f}")

# ------------------------------
# 캘리브레이션 실행
# ------------------------------
print("\n[INFO] 캘리브레이션 시작")
calibrate("정상", class_idx=0)
calibrate("졸음", class_idx=1)
print("[INFO] 캘리브레이션 종료!\n")

print("Drowsiness detection")

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    frame = cv2.flip(frame, 1)
    original = frame.copy()

    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        x1, y1, x2, y2 = boxes[0].astype(int)
        face = original[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_tensor = transform(face)
        face_buffer.append(face_tensor)

        if len(face_buffer) == 5:
            input_seq = torch.stack(list(face_buffer)).unsqueeze(0).to(device)
            action_idx = action2idx.get(action_str, action2idx["기타"])
            action_vec = torch.nn.functional.one_hot(torch.tensor([action_idx]), num_classes=len(action_vocab)).float().to(device)

            with torch.no_grad():
                output = model(input_seq, action_vec)
                #output *= calibration_weights.unsqueeze(0)
                 # 🔥 여기 추가: Softmax 출력
                probs = torch.softmax(output, dim=1)
                print("📊 Softmax:", probs.cpu().numpy())

                pred = torch.argmax(probs, dim=1).item()
            

            label = "Drowsy" if pred == 1 else "Awake"
            color = (0, 0, 255) if pred == 1 else (0, 255, 0)
            cv2.rectangle(original, (x1, y1), (x2, y2), color, 2)
            cv2.putText(original, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("실시간 졸음 인식", original)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print("🔥 Raw output:", output)
    print("🔥 Predicted:", pred)



pipeline.stop()
cv2.destroyAllWindows()
print("🛑 종료됨")
