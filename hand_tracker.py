"""Hand tracking and gesture recognition module"""

import mediapipe as mp
import cv2
import numpy as np
from config import MEDIAPIPE_CONFIG, GESTURE_THRESHOLDS, HAND_LANDMARKS

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(**MEDIAPIPE_CONFIG)
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect_hands(self, frame):
        """Detect hands in frame and return results"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb_frame)
    
    def get_fingertip_position(self, hand_landmarks, frame_shape):
        """Get index fingertip position in pixel coordinates"""
        h, w, _ = frame_shape
        index_tip = hand_landmarks.landmark[8]  # Index fingertip
        return int(index_tip.x * w), int(index_tip.y * h)
    
    def is_finger_extended(self, hand_landmarks, finger_landmarks):
        """Check if a finger is extended using angle calculation"""
        base = hand_landmarks.landmark[finger_landmarks[0]]
        middle = hand_landmarks.landmark[finger_landmarks[1]]
        tip = hand_landmarks.landmark[finger_landmarks[3]]
        
        angle = self._calculate_angle(base, middle, tip)
        return angle > GESTURE_THRESHOLDS['finger_extension_angle']
    
    def detect_writing_gesture(self, hand_landmarks):
        """Detect writing gesture: index extended, others bent"""
        index_extended = self.is_finger_extended(hand_landmarks, HAND_LANDMARKS['INDEX'])
        middle_extended = self.is_finger_extended(hand_landmarks, HAND_LANDMARKS['MIDDLE'])
        pinky_extended = self.is_finger_extended(hand_landmarks, HAND_LANDMARKS['PINKY'])
        
        return index_extended and not middle_extended and not pinky_extended
    
    def detect_erasing_gesture(self, hand_landmarks):
        """Detect erasing gesture: all fingers extended and close"""
        fingers_extended = all([
            self.is_finger_extended(hand_landmarks, HAND_LANDMARKS['INDEX']),
            self.is_finger_extended(hand_landmarks, HAND_LANDMARKS['MIDDLE']),
            self.is_finger_extended(hand_landmarks, HAND_LANDMARKS['RING']),
            self.is_finger_extended(hand_landmarks, HAND_LANDMARKS['PINKY'])
        ])
        
        fingertips_close = self._are_fingertips_close(hand_landmarks)
        return fingers_extended and fingertips_close
    
    def draw_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks on frame"""
        self.mp_drawing.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
        )
    
    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle at point p2"""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return np.degrees(np.arccos(cos_angle))
    
    def are_fingertips_close(self, hand_landmarks, finger1_landmarks, finger2_landmarks):
        """Check if fingertips are close enough ="""
        base = hand_landmarks.landmark[finger1_landmarks[0]]
        tip1 = hand_landmarks.landmark[finger1_landmarks[3]]
        tip2 = hand_landmarks.landmark[finger2_landmarks[3]]
        angle = self._calculate_angle(base, tip1, tip2)
        return angle > GESTURE_THRESHOLDS['fingertip_close_angle']
        
        