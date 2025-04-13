import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import os
import time

class FirebaseUploader(Node):
    def __init__(self):
        super().__init__('firebase_uploader_node')
        self.subscription = self.create_subscription(
            String,
            '/face_identification',
            self.listener_callback,
            10
        )

        key_path = "/home/sienna/Workspace/dream/src/face_login_pkg/face_login_pkg/firebase-key.json"
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        self.get_logger().info("✅ Firebase 연동 완료!")

        # ✅ 중복 로그인 방지용
        self.last_user_id = None
        self.logged_in = False

        # ✅ 로그인된 사용자 퍼블리시용 토픽
        self.current_user_pub = self.create_publisher(String, '/current_user', 10)

    def listener_callback(self, msg):
        user_id = msg.data
        now = time.time()

        # ✅ 한 번 로그인했으면 더 이상 기록 안 함
        if self.logged_in and user_id == self.last_user_id:
            self.get_logger().info(f"⚠️ 이미 로그인된 사용자입니다: {user_id} → 기록 안 함")
            return

        self.last_user_id = user_id
        self.logged_in = True
        self.log_login(user_id)

    def log_login(self, user_id):
        log_ref = self.db.collection("users").document(user_id).collection("login_logs")
        log_ref.add({
            "timestamp": datetime.now().isoformat(),
            "source": "pc"
        })
        self.get_logger().info(f"✅ 로그인 기록 저장됨: {user_id}")
        
        # ✅ current_user 토픽 퍼블리시
        self.current_user_pub.publish(String(data=user_id))
        self.get_logger().info(f"📤 current_user 토픽 발행됨: {user_id}")

def main(args=None):
    rclpy.init(args=args)
    node = FirebaseUploader()
    rclpy.spin(node)
    rclpy.shutdown()
