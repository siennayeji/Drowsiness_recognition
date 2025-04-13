import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
import numpy as np

from .eye_detector import EyeDetector
from .yawn_detector import YawnDetector
from .nodding_detector import NoddingDetector

class DrowsinessDetectionNode(Node):
    def __init__(self):
        super().__init__('drowsiness_detection_node')
        self.subscription = self.create_subscription(
            Float32MultiArray, 
            '/face/landmarks', 
            self.drowsiness_detection_callback, 
            10)
        self.publisher = self.create_publisher(
            String, 
            '/drowsiness/status', 
            10)
        
        self.eye_detector = EyeDetector()
        self.yawn_detector = YawnDetector()
        self.nodding_detector = NoddingDetector()

        self.eye_closed_start_time = None
        self.eye_closed_duration_threshold = 2.0 

        self.yawn_start_time = None
        self.yawn_duration_threshold = 3

        self.get_logger().info("=============== Drowsiness Detection Node Started ===============")

    # -----------------------------
    # 1) 눈 감김 시간 로직 함수
    # -----------------------------
    def check_eyes_closed(self, ear_avg):
        """
        EAR < eye_threshold 상태가 2초 이상 유지되면 True 반환
        그렇지 않으면 False
        """
        if ear_avg < self.eye_detector.eye_threshold:
            if self.eye_closed_start_time is None :
                self.eye_closed_start_time = time.time()
            else:
                duration = time.time() - self.eye_closed_start_time
                if duration >= self.eye_closed_duration_threshold:
                    return True
        else:
            self.eye_closed_start_time = None
        return False
    
    # ----------------------------
    # 2) 하품 시간 로직 함수
    # -----------------------------
    def check_yawning(self, mar_avg):
        """
        MAR > yawn_detector.threshold 상태가 3초 이상 유지되면 True
        그렇지 않으면 False
        """
        if mar_avg > self.yawn_detector.threshold:
            if self.yawn_start_time is None :
                self.yawn_start_time = time.time()
            else:
                duration = time.time() - self.yawn_start_time 
                if duration >=self.yawn_duration_threshold:
                    return True
        else:
            self.yawn_start_time = None
        return False
    
    # -----------------------------
    # 3) 메인 콜백: 졸음 인식
    # -----------------------------
    def drowsiness_detection_callback(self, msg):
        landmarks = np.array(msg.data).reshape(-1, 2)

        ear_avg = self.eye_detector.detect(landmarks)
        mar_avg = self.yawn_detector.detect(landmarks)
        nodding_status = self.nodding_detector.detect(landmarks)

        self.yawn_detector.calibrate_mouth(mar_avg)
        self.eye_detector.calibrate_eyes(ear_avg)

        if (not self.yawn_detector.calibrated) or (not self.eye_detector.eye_calibrated):
            status = "Calibrating ..."
            self.publisher.publish(String(data=status))
            return 
        
        eyes_closed = self.check_eyes_closed(ear_avg)
        yawning = self.check_yawning(mar_avg)
        
        if eyes_closed and nodding_status == "Nodding" and yawning:
            status = "하품 감지"
        elif eyes_closed and yawning:
            status = "하품 감지"
        elif nodding_status == "Nodding" and yawning:
            status = "하품 감지"
        elif yawning:
            status = "하품 감지"
            
        elif eyes_closed and nodding_status == "Nodding":
            status = "졸음 감지 (앞) / 눈 감김 감지 "
        elif eyes_closed and nodding_status == "Nodding_side":
            status = "졸음 감지 (좌우) / 눈 감김 감지 "
        elif eyes_closed:
            status = "눈 감김 감지"

        else:
            status = "Normal"

        self.publisher.publish(String(data=status))

def main(args=None):
    rclpy.init(args=args)
    node = DrowsinessDetectionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
