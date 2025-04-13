import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

class DrowsinessLogger(Node):
    def __init__(self):
        super().__init__('drowsiness_logger_node')
        
        self.current_user = None

        # 구독자 생성
        self.create_subscription(String, '/current_user', self.user_callback, 10)
        self.create_subscription(String, '/drowsiness/status', self.drowsiness_callback, 10)

        # Firebase 연결
        key_path = "/home/sienna/Workspace/dream/src/face_login_pkg/face_login_pkg/firebase-key.json"
        cred = credentials.Certificate(key_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()

        self.get_logger().info("😴 졸음 기록 노드 시작됨!")

    def user_callback(self, msg):
        self.current_user = msg.data
        self.get_logger().info(f"✅ 로그인된 사용자 수신: {self.current_user}")
    
    def drowsiness_callback(self, msg):
        if self.current_user is None:
            self.get_logger().warn("⚠️ 사용자 정보 없음 → 졸음 기록 안 함")
            return

        # 🔍 메시지 해석 → 이벤트 분류
        text = msg.data
        event = None
        if "하품" in text:
            event = "yawn"
        elif "눈 감김" in text:
            event = "eye_closed"
        elif "졸음" in text:
            event = "drowsy"

        if event:
            log_ref = self.db.collection("users").document(self.current_user).collection("drowsiness_logs")
            log_ref.add({
                "timestamp": datetime.now().isoformat(),
                "event": event,
                "source": "ros2"
            })
            self.get_logger().info(f"📝 졸음 로그 저장됨: {event}")
        else:
            self.get_logger().info(f"ℹ️ 저장 안 함: {text}")
def main(args=None):
    rclpy.init(args=args)
    node = DrowsinessLogger()
    rclpy.spin(node)
    rclpy.shutdown()
