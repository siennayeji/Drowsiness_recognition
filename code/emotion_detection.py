# -*- coding: utf-8 -*-

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, models

# ✅ 감정 레이블 정의
EMOTION_LABELS = {
    0: "기쁨",
    1: "분노",
    2: "놀람",
    3: "슬픔",
    4: "중립"
}

# ✅ 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 5)  # 클래스 수 5개
model.load_state_dict(torch.load("emotion_recognition_model_small.pth", map_location=device))
model = model.to(device)
model.eval()

# ✅ 이미지 전처리 정의
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ✅ 웹캠 열기
cap = cv2.VideoCapture(0)  # 0번 카메라

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("🟢 실시간 감정 인식 시작!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 프레임 복사본 만들기 (출력용)
    output_frame = frame.copy()

    # BGR → RGB 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 전처리
    input_tensor = transform(rgb_frame)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # 배치 차원 추가

    # 예측
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        emotion = EMOTION_LABELS[pred]
        confidence = probs[0][pred].item()

    # 결과 표시
    label = f"{emotion} ({confidence*100:.1f}%)"
    cv2.putText(output_frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Recognition', output_frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ 정리
cap.release()
cv2.destroyAllWindows()
