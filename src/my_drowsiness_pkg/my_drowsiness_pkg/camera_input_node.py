# my_drowsiness_pkg/camera_input_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class CameraInputNode(Node):
    def __init__(self):
        super().__init__('camera_input_node')
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.timer = self.create_timer(0.033, self.timer_callback)  # ~30FPS
        self.cap = cv2.VideoCapture(0)
        self.bridge = CvBridge()
        self.get_logger().info('Camera input node started.')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraInputNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
