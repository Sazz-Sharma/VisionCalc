
import cv2
import numpy as np
from config import DRAWING_CONFIG

class DrawingCanvas:
    def __init__(self, frame_shape):
        self.canvas = np.zeros(frame_shape, dtype=np.uint8)
        self.prev_x, self.prev_y = None, None
        
    def draw_line(self, x, y):
        """Draw line from previous position to current position"""
        if self.prev_x is not None and self.prev_y is not None:
            cv2.line(
                self.canvas, 
                (self.prev_x, self.prev_y), 
                (x, y), 
                DRAWING_CONFIG['pen_color'], 
                DRAWING_CONFIG['pen_thickness']
            )
        self.prev_x, self.prev_y = x, y
    
    def erase_at(self, x, y):
        """Erase at given position"""
        cv2.circle(
            self.canvas, 
            (x, y), 
            DRAWING_CONFIG['eraser_size'], 
            (0, 0, 0), 
            -1
        )
        self.prev_x, self.prev_y = x, y
    
    def stop_drawing(self):
        """Reset drawing state"""
        self.prev_x, self.prev_y = None, None
    
    def clear(self):
        """Clear the entire canvas"""
        self.canvas.fill(0)
        self.stop_drawing()
    
    def get_combined_view(self, frame):
        """Combine frame with canvas overlay"""
        return cv2.addWeighted(
            frame, 
            DRAWING_CONFIG['frame_opacity'], 
            self.canvas, 
            DRAWING_CONFIG['canvas_opacity'], 
            0
        )