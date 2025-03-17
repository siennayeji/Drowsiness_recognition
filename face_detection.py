import cv2
import dlib
import pyrealsense2 as rs
import numpy as np

# Dlib 얼굴 검출기 로드
detector = dlib.get_frontal_face_detector()

# RealSense 설정 (RGB 스트림만 사용)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB 스트림만 활성화
pipeline.start(config)

while True:
    # RGB 프레임 가져오기
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        continue

    # RGB 이미지 변환
    color_image = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = detector(gray)

    # 검출된 얼굴에 박스 그리기
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 화면에 출력
    cv2.imshow("Dlib Face Detection (RGBD Camera)", color_image)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 종료
pipeline.stop()
cv2.destroyAllWindows()
