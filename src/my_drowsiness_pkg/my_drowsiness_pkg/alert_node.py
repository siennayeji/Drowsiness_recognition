# my_drowsiness_pkg/alert_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

class AlertNode(Node):
    def __init__(self):
        super().__init__('alert_node')
        self.subscription = self.create_subscription(Int32, '/drowsiness/state', self.callback, 10)
        self.get_logger().info('Alert node started.')

    def callback(self, msg):
        if msg.data == 1:
            self.get_logger().warn('⚠️ Drowsiness detected! Wake up!')
        else:
            self.get_logger().info('Normal state.')

def main(args=None):
    rclpy.init(args=args)
    node = AlertNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
