import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import random
import time

class EmotionReactionNode(Node):
    def __init__(self):
        super().__init__('emotion_reaction_node')
        self.subscription = self.create_subscription(
            String,
            '/emotion/status',
            self.listener_callback,
            10
        )
        self.publisher = self.create_publisher(String, '/emotion/message', 10)

        self.current_emotion = None
        self.emotion_start_time = None
        self.duration_threshold = 5.0  # 5초 이상 유지 시 반응

        self.last_trigger_time = 0
        self.trigger_interval = 10  # 메시지 발행 간 최소 간격 (초)

        # 감정별 메시지
        self.messages = {
            "happy": [
                "오늘 기분 정말 좋아 보여요! 😊",
                "행복한 기분 오래 유지해요!",
                "좋은 일 가득하길 바랄게요 💛"
            ],
            "sad": [
                "괜찮아요, 오늘은 충분히 잘하고 있어요.",
                "힘들 땐 잠시 쉬어가도 괜찮아요.",
                "당신은 혼자가 아니에요 💙"
            ],
            "hurt": [
                "마음 아팠던 순간, 이겨낼 수 있어요.",
                "당신은 생각보다 훨씬 강한 사람이에요.",
                "지금도 잘 버티고 있어요. 힘내요!"
            ],
            "surprise": [
                "놀라운 일이 있었나 봐요! 😮",
                "기대하지 못한 순간이 찾아왔군요!",
                "가끔 놀람은 일상의 활력소가 되기도 해요 ✨"
            ],
            "angry": [
                "화날 땐 잠시 멈추고 심호흡해봐요. 🍃",
                "감정을 억누르지 말고 천천히 다독여줘요.",
                "당신의 마음이 편안해지기를 바라요 🙏"
            ]
        }

    def listener_callback(self, msg):
        content = msg.data.split()[0]  # "happy (99.9%)" → "happy"
        now = time.time()

        if content != self.current_emotion:
            self.current_emotion = content
            self.emotion_start_time = now
            return

        # 감정이 일정 시간 이상 유지되었고, 마지막 트리거 이후 충분히 시간이 지났으면
        if now - self.emotion_start_time >= self.duration_threshold and now - self.last_trigger_time >= self.trigger_interval:
            self.last_trigger_time = now
            if content in self.messages:
                message = random.choice(self.messages[content])
                msg_out = String()
                msg_out.data = message
                self.publisher.publish(msg_out)
                self.get_logger().info(f"🟢 {content} 감정 유지 → 메시지 발행: {message}")

def main(args=None):
    rclpy.init(args=args)
    node = EmotionReactionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
