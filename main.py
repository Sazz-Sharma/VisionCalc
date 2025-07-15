import cv2
from Tracking.hand_tracker import HandTracker
from canvas import DrawingCanvas
from equation_extraction import Solver

class VisionCalc:
    def __init__(self):
        self.hand_tracker = HandTracker()
        self.canvas = None
        self.solver = Solver()
        self.cap = cv2.VideoCapture(0)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror effect
            if self.canvas is None:
                self.canvas = DrawingCanvas(frame.shape)
            
            results = self.hand_tracker.detect_hands(frame)
            if results.multi_hand_landmarks:
                self._process_hand_gestures(results, frame)
            
            combined = self.canvas.get_combined_view(frame)
            cv2.imshow('VisionCalc', combined)

            if self._handle_keyboard_input():
                break
            
        self._cleanup()

    def _process_hand_gestures(self, results, frame):
        for hand_landmarks in results.multi_hand_landmarks:
            x,y = self.hand_tracker.get_fingertip_position(hand_landmarks, frame.shape)
            is_writing = self.hand_tracker.detect_writing_gesture(hand_landmarks)
            is_erasing = self.hand_tracker.detect_erasing_gesture(hand_landmarks)

            if is_writing:
                self.canvas.draw_line(x, y)
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1) #green

            elif is_erasing:
                self.canvas.erase_at(x, y)
                cv2.circle(frame, (x, y), 30, (0, 255, 255), 2) #cyan
            
            else:
                self.canvas.stop_drawing()
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)  # Red 
            
            self.hand_tracker.draw_landmarks(frame, hand_landmarks)
        
    def _handle_keyboard_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            self.canvas.clear()
        elif key == ord('q'):
            return True
        elif key == ord('a'):
            self._analyze_equation()
        
        return False

    def _analyze_equation(self):
        print("Analyzing equation...") 
        equation_image = self.canvas.canvas
        print(equation_image)
        cv2.imshow('Equation Image', equation_image)
        cv2.waitKey(2000)
        cv2.destroyWindow('Equation Image')
        
        equation = self.solver.extract_equation(equation_image)

        if equation:
            solution = self.solver.solve_equation(equation)
            print(f"Equation: {equation}, Solution: {solution}")

            cv2.putText(
                self.canvas.canvas, 
                f"Solution: {solution}", 
                (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 255, 255), 
                2
            )
        else:
            print("No equation detected.")
            cv2.putText(
                self.canvas.canvas, 
                "No equation detected", 
                (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )
             

    def _cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    vision_calc = VisionCalc()
    vision_calc.run()
