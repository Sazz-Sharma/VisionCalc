from wolframalpha_wrapper import WolframAlphaWrapper
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import os
import cv2

load_dotenv()

WOLFRAM_API_KEY = os.getenv("WOLFRAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

wolfram = WolframAlphaWrapper(WOLFRAM_API_KEY, use_simple=True)
genai.configure(api_key=GEMINI_API_KEY)



class Solver:
    def __init__(self):
        self.vision_model = genai.GenerativeModel('gemini-1.5-flash')
        self.chat_model = genai.GenerativeModel('gemini-1.5-flash')
    
    def extract_equation(self, canvas):
        try:
            pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY))

            prompt = """ 
            Supporse you are a math expert specialized in extracting equation from this handwritten image.
            Please extract the equation from the image so that the meaning is preserved, and the other AI agent/WolframAlpha
            can use it to solve the equation. 
            For example, if the image contains ∫x^2 , you should return "integrate x^2 dx" or anything but should be clearly understable and interpretable by WolframAlpha.

            """

            response = self.vision_model.generate_content([prompt, pil_image])
            return response.text.strip()
        except Exception as e:
            print(f"Error extracting equation: {e}")
            return None 
    
    def solve_equation(self, equation):
        try:
            response = wolfram.query(equation, detailed=True)
            return response
        except Exception as e:
            print(f"Error solving equation: {e}")
            print("Trying with chat model...")
        try:
            prompt = f"""
            You are a math expert specialized in solving equations.
            Please solve the following equation and return the solution in a clear and concise manner.
            just return the solution without any additional text or explanation.
            {equation}
            """
            response = self.chat_model.generate_content([prompt])
            return response.text.strip()
        except Exception as e:
            print(f"Error solving equation with chat model: {e}")
            return None


