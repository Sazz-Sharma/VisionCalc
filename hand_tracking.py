import mediapipe as mp 
import cv2
import math
import numpy as np
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("Camera Initialized. Press 'q' to quit.")

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def total_distance(hand_landmarks, finger_landmarks):
    total_distance = 0
    for i in range(len(finger_landmarks) - 1):
        point1 = hand_landmarks.landmark[finger_landmarks[i]]
        point2 = hand_landmarks.landmark[finger_landmarks[i + 1]]
        total_distance += calculate_distance(point1, point2)
    return total_distance
    
def is_finger_extended(hand_landmarks, finger_landmarks):
    distance = total_distance(hand_landmarks, finger_landmarks)
    return distance > 0.15  # Threshold to determine if the finger is extended


def is_finger_extended_relative(hand_landmarks, finger_landmarks, debug=False, name=""):
    base = hand_landmarks.landmark[finger_landmarks[0]]
    middle = hand_landmarks.landmark[finger_landmarks[2]]
    tip = hand_landmarks.landmark[finger_landmarks[3]]

    base_to_middle = calculate_distance(base, middle)
    middle_to_tip = calculate_distance(middle, tip) 
    base_to_tip = calculate_distance(base, tip)

    expected_extended_length = base_to_middle + middle_to_tip
    actual_length = base_to_tip

    straightness_ratio = actual_length / expected_extended_length if expected_extended_length > 0 else 0
    if debug:
        print(f"Straightness {name}: {straightness_ratio:.2f}")
    # print(f"Straightness Ratio: {straightness_ratio:.2f}")
    return straightness_ratio > 0.97 # Adjust threshold as needed

def fingertips_distance(hand_landmarks, finger1_tip, finger2_tip):
    tip1 = hand_landmarks.landmark[finger1_tip]
    tip2 = hand_landmarks.landmark[finger2_tip]
    return calculate_distance(tip1, tip2)

def are_fingertips_close(hand_landmarks, finger1_tip, finger2_tip, threshold=0.05):
    distance = fingertips_distance(hand_landmarks, finger1_tip, finger2_tip)
    # print(f"Distance between fingertips: {distance:.2f}")
    return distance < threshold

def calculate_angle(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    cos_angle = dot_product / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure the value is within the valid range for arccos
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

def is_finger_extended_angle(hand_landmarks, finger_landmarks, debug=False, name=""):
    base = hand_landmarks.landmark[finger_landmarks[0]]
    middle = hand_landmarks.landmark[finger_landmarks[1]]
    tip = hand_landmarks.landmark[finger_landmarks[3]]

    angle = calculate_angle(base, middle, tip)
    
    if debug:
        print(f"Angle {name}: {angle:.2f} degrees")
    
    # Assuming an extended finger has an angle less than 45 degrees
    return angle >160  # Adjust threshold as needed

def are_fingertips_close_angle(hand_landmarks, finger1_landmarks, finger2_landmarks,debug= False, threshold=115):
    base = hand_landmarks.landmark[finger1_landmarks[0]]
    tip1 = hand_landmarks.landmark[finger1_landmarks[3]]
    tip2 = hand_landmarks.landmark[finger2_landmarks[3]]
    angle = calculate_angle(base, tip1, tip2)  # Using the base
    if debug:
        print(f"Angle between fingertips: {angle:.2f} degrees")
    return angle > threshold

canvas = None
prev_x, prev_y = None, None


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect
    if canvas is None:
        canvas = np.zeros_like(frame)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)
            

            # print(f"Index Tip: ({index_x}, {index_y}), Middle Tip: ({middle_x}, {middle_y})")


            index_extended = is_finger_extended_angle(hand_landmarks, [5, 6, 7, 8], debug=True, name="Index Finger")
            middle_extended = is_finger_extended_angle(hand_landmarks, [9, 10, 11, 12], debug=True, name="Middle Finger")
            thumb_extended = is_finger_extended_angle(hand_landmarks, [2, 3, 1, 4], debug=True, name="Thumb")
            ring_extended = is_finger_extended_angle(hand_landmarks, [13, 14, 15, 16], debug=True, name="Ring Finger")
            pinky_extended = is_finger_extended_angle(hand_landmarks, [17, 18, 19, 20], debug=True, name="Pinky Finger")

            # print(f"Index Total Distance : {total_distance(hand_landmarks, [5, 6, 7, 8])} ")
            # print(f"Middle Total Distance : {total_distance(hand_landmarks, [9, 10, 11, 12])} ")
            # print(f"Index Finger Extended: {index_extended}, Middle Finger Extended: {middle_extended}")
            print(f"Ring Extended: {ring_extended}, Pinky Extended: {pinky_extended}, Middle Extended: {middle_extended}, Index Extended: {index_extended}")
            is_writing = index_extended and not middle_extended and not pinky_extended
            # is_index_middle_close = are_fingertips_close(hand_landmarks, 8, 12)  # Index and Middle tips

            is_index_middle_close = are_fingertips_close_angle(hand_landmarks, [5, 6, 7, 8], [9, 10, 11, 12], debug= True)  # Index and Middle tips

            is_erasing = index_extended and middle_extended and ring_extended and pinky_extended and is_index_middle_close

            if is_writing:
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), (255, 0, 0), 3)
                prev_x, prev_y = index_x, index_y

                color = (0, 255, 0) if index_extended else (0, 0, 255)
                cv2.circle(frame, (index_x, index_y), 10, color, -1)

            elif is_erasing:
                eraser_size = 30
                if prev_x is not None and prev_y is not None:
                    cv2.circle(canvas, (prev_x, prev_y), eraser_size, (0, 0, 0), -1)
                prev_x, prev_y = index_x, index_y
                
                cv2.circle(frame, (index_x, index_y), eraser_size, (0, 255, 255), 2)

            else:
                prev_x, prev_y = None, None
                cv2.circle(frame, (index_x, index_y), 10, (0, 0, 255), -1)

            

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    combined = cv2.addWeighted(frame,0.7, canvas, 0.7 ,0)

    cv2.imshow('Hand Tracking', combined)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
        print("Canvas cleared.")

cap.release()
cv2.destroyAllWindows()


