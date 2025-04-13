import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from collections import deque
import base64
import cv2
from cv_bridge import CvBridge
import json

class SequenceBufferNode(Node):
    def __init__(self):
        super().__init__('sequence_buffer_node')
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.listener_callback, 10)
        self.publisher = self.create_publisher(Image, '/sequence/image_sequence', 10)
        self.bridge = CvBridge()
        self.buffer = deque(maxlen=5)  # 시퀀스 길이 T = 5
        self.get_logger().info('Sequence buffer node started.')

    def listener_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        _, buffer = cv2.imencode('.jpg', img)
        encoded = base64.b64encode(buffer).decode('utf-8')
        self.buffer.append(encoded)
        if len(self.buffer) == self.buffer.maxlen:
            json_msg = Int32()
            json_msg.data = json.dumps({'images': list(self.buffer)})
            self.publisher.publish(json_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SequenceBufferNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()