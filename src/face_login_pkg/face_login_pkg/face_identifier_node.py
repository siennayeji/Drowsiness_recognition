import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
from keras_facenet import FaceNet
import firebase_admin
from firebase_admin import credentials, firestore
from scipy.spatial.distance import cosine


class FaceIdentifierNode(Node):
    def __init__(self):
        super().__init__('face_identifier_node')
        self.bridge = CvBridge()
        self.detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)
        self.embedder = FaceNet()

        # Firebase 초기화
        cred_path = "/home/sienna/Workspace/dream/src/face_login_pkg/face_login_pkg/firebase-key.json"
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()

        self.current_email = None
        self.target_embedding = None
        self.authenticated = False

        self.create_subscription(String, '/current_email', self.email_callback, 10)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.publisher_ = self.create_publisher(Bool, '/user_authenticated', 10)

        self.get_logger().info("🔐 얼굴 인증 노드 실행 중...")

    def email_callback(self, msg):
        full_email = msg.data.strip()

        if not full_email.startswith("[drowsiness_detection]"):
            self.get_logger().info("⚠️ [drowsiness_detection] prefix 없음 → 무시됨")
            return

        self.current_email = full_email.replace("[drowsiness_detection]", "").strip().lower()
        self.get_logger().info(f"📩 인증용 이메일 수신: {self.current_email}")

        # Firestore에서 해당 사용자의 임베딩 로드
        try:
            doc = self.db.collection("users").document(self.current_email).collection("face_embedding").document("vector").get()
            if doc.exists:
                self.target_embedding = np.array(doc.to_dict()["embedding"])
                self.authenticated = False
                self.get_logger().info("✅ 사용자 임베딩 불러오기 성공")
            else:
                self.get_logger().warn("❌ 등록된 얼굴 임베딩 없음")
        except Exception as e:
            self.get_logger().error(f"Firebase 오류: {e}")

    def image_callback(self, msg):
        if self.target_embedding is None or self.authenticated:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.detections:
            for det in results.detections:
                h, w, _ = frame.shape
                bbox = det.location_data.relative_bounding_box
                x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                x2, y2 = x1 + int(bbox.width * w), y1 + int(bbox.height * h)
                face_img = frame[y1:y2, x1:x2]

                if face_img.size == 0:
                    continue

                emb = self.embedder.embeddings([face_img])[0]
                similarity = 1 - cosine(self.target_embedding, emb)

                self.get_logger().info(f"🔍 유사도: {similarity:.3f}")

                if similarity >= 0.5:
                    self.get_logger().info("🎉 얼굴 인증 성공!")
                    self.publisher_.publish(Bool(data=True))
                    self.authenticated = True
                else:
                    self.get_logger().warn("😕 얼굴 인증 실패")

                return


def main(args=None):
    rclpy.init(args=args)
    node = FaceIdentifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()
