import cv2
import mediapipe as mp
import numpy as np
import json
import os
from keras_facenet import FaceNet
from datetime import datetime
import uuid

# 모델 로드
embedder = FaceNet()
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)

# 사용자 입력
name = input("이름을 입력하세요: ")
user_id = f"user_{uuid.uuid4().hex[:6]}"  # 고유 ID

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
user_dir = os.path.join(BASE_DIR, "users", f"{name}_{user_id}")
os.makedirs(user_dir, exist_ok=True)
emb_path = os.path.join(BASE_DIR, "user_embeddings.json")

# 카메라 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

saved = False
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
            x2, y2 = x1 + int(bbox.width * w), y1 + int(bbox.height * h)
            face_img = frame[y1:y2, x1:x2]

            if face_img.size == 0:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Press [S] to save", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and not saved:
                embedding = embedder.embeddings([face_img])[0]

                data = {}
                if os.path.exists(emb_path):
                    with open(emb_path, 'r') as f:
                        data = json.load(f)
                data[user_id] = {
                    "name": name,
                    "embedding": embedding.tolist()
                }
                with open(emb_path, 'w') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                img_path = os.path.join(user_dir, "face.jpg")
                cv2.imwrite(img_path, face_img)

                print(f"✅ {name} 님 등록 완료! ID: {user_id}")
                saved = True
                break

    cv2.imshow("Face Register", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
