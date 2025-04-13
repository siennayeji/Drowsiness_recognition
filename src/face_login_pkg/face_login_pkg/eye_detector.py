#!/usr/bin/env python3
import numpy as np
from scipy.spatial import distance as dist

class EyeDetector:
    def __init__(self, calibration_frames=300, k_eye=0.5):
        self.calibration_frames = calibration_frames
        self.k_eye = k_eye
        self.ear_values = []
        self.eye_calibrated = False
        self.baseline_ear = 0
        self.eye_threshold = 0

    def detect(self, landmarks):
        def compute_EAR(eye):
            A = dist.euclidean(eye[1], eye[5])
            B = dist.euclidean(eye[2], eye[4])
            C = dist.euclidean(eye[0], eye[3])
            ear = (A + B) / (2.0 * C)
            return ear

        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        left_EAR = compute_EAR(left_eye)
        right_EAR = compute_EAR(right_eye)
        ear_avg = (left_EAR + right_EAR) / 2.0
        return ear_avg

    def calibrate_eyes(self, ear_avg):
        if len(self.ear_values) < self.calibration_frames:
            self.ear_values.append(ear_avg)
        else:
            if not self.eye_calibrated:
                self.baseline_ear = np.mean(self.ear_values)
                std_ear = np.std(self.ear_values)
            
                self.eye_threshold = self.baseline_ear - self.k_eye * std_ear
                self.eye_calibrated = True
                print(f"ye Calibration done: Baseline EAR = {self.baseline_ear:.3f}, EAR Threshold = {self.eye_threshold:.3f}")
        return self.baseline_ear, self.eye_threshold