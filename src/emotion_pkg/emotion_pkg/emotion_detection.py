# -*- coding: utf-8 -*-

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import mediapipe as mp
import torch.nn as nn

# ✅ 감정 레이블 정의
EMOTION_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

# ✅ SimpleCNN 모델 정의
class DeepCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(DeepCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
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
        x = self.classifier(x)
        return x

# ✅ 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepCNN(num_classes=7)
model.load_state_dict(torch.load("deep_cnn_fer2013.pth", map_location=device))
model = model.to(device)
model.eval()

# ✅ 이미지 전처리 정의
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ✅ Mediapipe 얼굴 검출 준비
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# ✅ 웹캠 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("🟢 실시간 얼굴 인식 + 감정 인식 시작!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_detection.process(frame_rgb)

    output_frame = frame.copy()

    if result.detections:
        for detection in result.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
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

            # 얼굴 부분 전처리
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_gray = np.expand_dims(face_gray, axis=2)
            input_tensor = transform(face_gray)
            input_tensor = input_tensor.unsqueeze(0).to(device)

            # 감정 예측
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                emotion = EMOTION_LABELS[pred]
                confidence = probs[0][pred].item()

            # 얼굴 박스와 감정 라벨 표시
            cv2.rectangle(output_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"{emotion} ({confidence*100:.1f}%)"
            cv2.putText(output_frame, label, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 결과 화면 출력
    cv2.imshow('Emotion Recognition', output_frame)

    # 'q' 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ 정리
cap.release()
cv2.destroyAllWindows()
face_detection.close()
