import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from model_train import DrowsinessModel  # 모델 클래스 불러오기

# ✅ RealSense 카메라 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# ✅ 얼굴 검출을 위한 OpenCV Haar Cascade 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ✅ 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DrowsinessModel().to(device)
model.load_state_dict(torch.load("drowsiness_model.pth", map_location=device))
model.eval()

# ✅ 이미지 전처리 변환
def transform_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0).to(device)

# ✅ 실시간 얼굴 & 졸음 감지 루프
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_tensor = transform_image(face_roi)
        
        # 모델 추론
        with torch.no_grad():
            output = model(face_tensor)
            _, pred = torch.max(output, 1)
        
        label = "Awake" if pred.item() == 2 else "Drowsy" if pred.item() == 1 else "Yawning"
        color = (0, 255, 0) if pred.item() == 2 else (0, 0, 255)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()