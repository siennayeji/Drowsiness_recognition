import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
import json
import os
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

class FaceIdentifierNode(Node):
    def __init__(self):
        super().__init__('face_identifier_node')
        self.publisher_ = self.create_publisher(String, '/face_identification', 10)
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        self.embedder = FaceNet()
        self.detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)
        self.bridge = CvBridge()

        self.user_data = self.load_embeddings()
        self.identified = False
        self.get_logger().info("✅ 얼굴 인식 노드 실행 중... 카메라 토픽 수신 대기 중")

    def load_embeddings(self):
        path = "/home/sienna/Workspace/dream/src/face_login_pkg/face_login_pkg/user_embeddings.json"
        with open(path, 'r') as f:
            return json.load(f)

    def image_callback(self, msg):
        if self.identified:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                x2, y2 = x1 + int(bbox.width * w), y1 + int(bbox.height * h)
                face_img = frame[y1:y2, x1:x2]

                if face_img.size == 0:
                    continue

                emb = self.embedder.embeddings([face_img])[0]
                min_dist = 1.0
                identified_user_id = "unknown"

                for user_id, info in self.user_data.items():
                    user_emb = np.array(info["embedding"])
                    dist = cosine(emb, user_emb)
                    if dist < 0.5 and dist < min_dist:
                        identified_user_id = user_id
                        min_dist = dist

                if identified_user_id != "unknown":
                    msg = String()
                    msg.data = identified_user_id
                    self.publisher_.publish(msg)
                    self.get_logger().info(f"✅ 사용자 인식됨: {identified_user_id}")

                    self.identified = True
                    # 얼굴 인식 종료 로그 추가
                    self.get_logger().info("👋 얼굴 인식 완료. 노드 계속 실행 중.")
                    return

        # 시각화도 하고 싶다면 필요시 이미지 띄우기 가능
        cv2.imshow("Face Identifier", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = FaceIdentifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()
