import os
import requests
from moviepy.editor import VideoFileClip
import whisper
import cv2
import base64
from io import BytesIO
from PIL import Image

class AudioExtractor:
    """
    Extracts audio from a video file and saves it as a .wav file.
    """
    def __init__(self, video_path):
        self.video_path = video_path
        self.audio_path = os.path.splitext(video_path)[0] + '.wav'

    def extract(self):
        """
        Extracts the audio track from the video.

        :return: True if extraction is successful, False otherwise.
        """
        if not os.path.exists(self.video_path):
            print(f"Error: Video not found at {self.video_path}")
            return False
        try:
            with VideoFileClip(self.video_path) as clip:
                if clip.audio is None:
                    print("No audio track found.")
                    return False
                clip.audio.write_audiofile(self.audio_path, codec='pcm_s16le', fps=16000)
                print(f"Audio extracted to {self.audio_path}")
                return True
        except Exception as e:
            print(f"Error during audio extraction: {e}")
            return False

def transcribe_audio(audio_path, model_name="base"):
    """
    Transcribes an audio file using OpenAI's Whisper model.

    :param audio_path: Path to the audio file.
    :param model_name: The name of the Whisper model to use (e.g., "base", "small", "medium").
    :return: The transcribed text.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    print("Transcribing audio...")
    result = model.transcribe(audio_path)
    return result.get("text", "Transcription failed.")

class VideoDescriber:
    """
    Describes video frames using a local LLaVA model via Ollama.
    """
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

    def _generate_description_llava(self, image: Image.Image, prompt: str) -> str:
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
                print(f"Processing frame {saved_frame_index} with LLaVA...")
                image_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                
                prompt = "Describe this scene in one concise sentence."
                description = self._generate_description_llava(image_rgb, prompt)
                
                descriptions.append({
                    "frame_index": saved_frame_index,
                    "timestamp_sec": round(frame_count / fps, 2) if fps > 0 else 0,
                    "description": description.strip()
                })
                saved_frame_index += 1

            frame_count += 1
        
        vidcap.release()
        return descriptions

def get_gemma_summary(transcription, video_descriptions):
    """
    Generates a summary of the video content using the Gemma model in Ollama.

    :param transcription: The audio transcription of the video.
    :param video_descriptions: A list of frame-by-frame descriptions.
    :return: The summary generated by Gemma.
    """
    print("\nRequesting summary from Gemma...")
    url = "http://localhost:11434/api/generate"
    
    # Construct a detailed prompt for Gemma
    prompt = (
        "You are a helpful assistant that describes video content chronologically.\n"
    "Based on the following frame-by-frame visual descriptions, which are already in order, "
    "describe the sequence of events in the video from start to finish.\n\n"
        "--- Audio Transcription ---\n"
        f"{transcription}\n\n"
        "--- Visual Descriptions ---\n"
    )
    for item in video_descriptions:
        prompt += f"- At {item['timestamp_sec']:.2f} seconds: {item['description']}\n"

    prompt += "\n--- Summary ---\n"

    payload = {
        "model": "gemma3:4b",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "Error: No response field in JSON output.")
    except requests.exceptions.RequestException as e:
        return f"Error: API request to Gemma failed - {e}"


if __name__ == "__main__":
    video_path = 'data/timer.mp4'  # Replace with your video file path
    descriptions_per_second = 0.5  # Describe one frame every 2 seconds

    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
    else:
        # --- 1. Audio Transcription ---
        audio_extractor = AudioExtractor(video_path)
        transcription = ""
        if audio_extractor.extract():
            try:
                transcription = transcribe_audio(audio_extractor.audio_path)
                print("\n--- Transcription (Whisper) ---")
                print(transcription)
            except Exception as e:
                print(f"An error occurred during transcription: {e}")

        # --- 2. Video Description ---
        video_describer = VideoDescriber(video_path, frame_rate=descriptions_per_second)
        video_descriptions = video_describer.describe_video()
        
        print("\n--- Video Descriptions (LLaVA) ---")
        if video_descriptions:
            for item in video_descriptions:
                print(f"[{item['frame_index']}] (Time: {item['timestamp_sec']}s) {item['description']}")
        else:
            print("No descriptions were generated.")

        # --- 3. Combined Summary with Gemma ---
        if transcription or video_descriptions:
            gemma_summary = get_gemma_summary(transcription, video_descriptions)
            print("\n--- Combined Summary (Gemma) ---")
            print(gemma_summary)
        else:
            print("\nCould not generate a summary as no transcription or descriptions were available.")