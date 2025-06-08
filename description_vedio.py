import cv2
import base64
import requests
from io import BytesIO
from PIL import Image
import os

class VideoDescriber:
    def __init__(self, video_path, frame_rate=1):
        """
        Initializes the VideoDescriber.

        :param video_path: Path to the input video file.
        :param frame_rate: How many frames per second to describe.
        """
        self.video_path = video_path
        self.frame_rate = frame_rate

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """Encodes a PIL Image to a base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _generate_description(self, image: Image.Image, prompt: str) -> str:
        """
        Generates a description for an image using a local LLaVA model via Ollama.
        """
        img_base64 = self._encode_image(image)
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llava",
            "prompt": prompt,
            "images": [img_base64],
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status() # Raise an exception for bad status codes
            return response.json().get("response", "Error: No response field in JSON output.")
        except requests.exceptions.RequestException as e:
            return f"Error: API request failed - {e}"

    def describe_video(self):
        """
        Iterates through video frames, generates a description for each selected frame,
        and returns a list of descriptions with timestamps.
        """
        if not os.path.exists(self.video_path):
            print(f"Error: Video file not found at {self.video_path}")
            return []

        vidcap = cv2.VideoCapture(self.video_path)
        if not vidcap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return []

        descriptions = []
        frame_count = 0
        saved_frame_index = 0
        
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(fps / self.frame_rate)) if fps > 0 else 1

        while True:
            success, frame_bgr = vidcap.read()
            if not success:
                break # End of video

            if frame_count % interval == 0:
                print(f"Processing frame {saved_frame_index}...")
                image_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                
                prompt = "Describe this scene in one concise sentence."
                description = self._generate_description(image_rgb, prompt)
                
                descriptions.append({
                    "frame_index": saved_frame_index,
                    "timestamp_sec": round(frame_count / fps, 2) if fps > 0 else 0,
                    "description": description.strip()
                })
                saved_frame_index += 1

            frame_count += 1
        
        vidcap.release()
        return descriptions

if __name__ == "__main__":
    video_file = "data/timer.mp4"
    descriptions_per_second = 0.5  # Describe one frame every 2 seconds

    if not os.path.exists(video_file):
        print(f"Video file not found: {video_file}")
    else:
        video_describer = VideoDescriber(video_file, frame_rate=descriptions_per_second)
        video_descriptions = video_describer.describe_video()
        
        print("\n--- Video Descriptions ---")
        if video_descriptions:
            for item in video_descriptions:
                print(f"[{item['frame_index']}] (Time: {item['timestamp_sec']}s) {item['description']}")
        else:
            print("No descriptions were generated.")