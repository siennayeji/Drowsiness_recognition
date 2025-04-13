import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import time

class UsbCameraNode(Node):
    def __init__(self):
        super().__init__('usb_camera_node')
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)

        # 카메라 설정 (해상도와 FPS 강제 설정)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 필요 시 해상도 조절
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # 30FPS로 설정

        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(f"✅ Camera FPS set to: {actual_fps}")

        if not self.cap.isOpened():
            self.get_logger().error("-------------Unable to open USB camera-------------")
            raise RuntimeError("Failed to open USB camera")

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("⚠️ Failed to capture frame from USB camera")
            return

        # BGR -> RGB 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # OpenCV 이미지를 ROS Image 메시지로 변환 후 퍼블리시
        ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="rgb8")
        self.publisher.publish(ros_image)

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init()
    node = UsbCameraNode()

    rate = node.create_rate(30)  # 30FPS로 설정

    try:
        while rclpy.ok():
            start_time = time.time()

            node.publish_frame()
            rclpy.spin_once(node, timeout_sec=0)  # Non-blocking spin
            elapsed_time = time.time() - start_time

            sleep_time = max(0, (1/30) - elapsed_time)  # 30FPS 유지
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node")
    finally:
        node.release()
        rclpy.shutdown()

if __name__ == "__main__":
    main()