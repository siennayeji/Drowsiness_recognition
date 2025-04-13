import cv2
import mediapipe as mp
import numpy as np
import json
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
import os

# 모델
embedder = FaceNet()
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)

# 사용자 임베딩 불러오기
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "user_embeddings.json"), 'r') as f:
    user_data = json.load(f)

# 실시간 인식
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

print("[INFO] 얼굴 인식 시작! 'q'를 눌러 종료하세요.")
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

            emb = embedder.embeddings([face_img])[0]
            name = "Unknown"
            min_dist = 1.0

            for user_id, info in user_data.items():
                user_name = info["name"]
                user_emb = np.array(info["embedding"])
                dist = cosine(emb, user_emb)

                if dist < 0.5 and dist < min_dist:
                    name = f"{user_name} ({user_id})"
                    min_dist = dist

            # 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face Identifier", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
