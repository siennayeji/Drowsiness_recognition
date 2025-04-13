import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import dlib
import numpy as np
import os

class FaceDetectorNode(Node):
    def __init__(self):
        super().__init__('face_detector_node')
        self.subscription = self.create_subscription(
            Image, 
            # '/camera/camera/color/image_raw', # Using RGBD Camera
            '/camera/image_raw',
            self.image_callback, 
            10)
        self.publisher = self.create_publisher(
            Float32MultiArray, 
            '/face/landmarks', 
            10)
        
        self.bridge = CvBridge()
        self.detector = dlib.get_frontal_face_detector()
        model_path = "/home/sienna/Workspace/dream/src/face_login_pkg/face_login_pkg/shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(model_path)
        self.get_logger().info("---------------------Face Detector Node Started------------------")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            return
        
        shape = self.predictor(gray, faces[0])
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

        # ROS 메시지 변환
        msg = Float32MultiArray()
        msg.data = landmarks.astype(np.float32).flatten().tolist()
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectorNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
