
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,
    'max_num_hands': 1,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}


GESTURE_THRESHOLDS = {
    'finger_extension_angle': 160,
    'fingertip_close_distance': 0.05,
    'fingertip_close_angle': 115
}


DRAWING_CONFIG = {
    'pen_color': (255, 255, 255),  # Blue
    'pen_thickness': 3,
    'eraser_size': 30,
    'canvas_opacity': 0.7,
    'frame_opacity': 0.7
}


HAND_LANDMARKS = {
    'INDEX': [5, 6, 7, 8],
    'MIDDLE': [9, 10, 11, 12],
    'RING': [13, 14, 15, 16],
    'PINKY': [17, 18, 19, 20],
    'THUMB': [1, 2, 3, 4]
}