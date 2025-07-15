import cv2
import numpy as np
from equation_extraction import Solver

solver = Solver()

def test_with_canvas():
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(canvas, "x^2 + 2x + 1 = 0", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Canvas", canvas)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    print("Testing equation extraction with canvas...")
    equation = solver.extract_equation(canvas)
    print(f"Extracted Equation: {equation}")

    if equation:
        print("Testing equation solving...")
        solution = solver.solve_equation(equation)
        print(f"Solution: {solution}")
    else:
        print("No equation extracted.")
    
test_with_canvas()