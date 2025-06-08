import os
import requests
from moviepy.editor import VideoFileClip
import whisper
import cv2
import base64
from io import BytesIO
from PIL import Image

class AudioExtractor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.audio_path = os.path.splitext(video_path)[0] + '.wav'

    def extract(self):
        if not os.path.exists(self.video_path):
            print(f"Error: Video not found at {self.video_path}")
            return False
        try:
            with VideoFileClip(self.video_path) as clip:
                if clip.audio is None:
                    print("No audio track found.")
                    return False
                clip.audio.write_audiofile(self.audio_path, codec='pcm_s16le', fps=16000, logger=None)
                print(f"Audio extracted to {self.audio_path}")
                return True
        except Exception as e:
            print(f"Error during audio extraction: {e}")
            return False

def transcribe_audio(audio_path, model_name="base"):
    if not os.path.exists(audio_path):
        return "No audio file found to transcribe."
    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    print("Transcribing audio... (This may take a moment)")
    result = model.transcribe(audio_path)
    return result.get("text", "Transcription failed.")

class VideoDescriber:
    def __init__(self, video_path, frame_rate=0.5):
        self.video_path = video_path
        self.frame_rate = frame_rate

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _generate_description_llava(self, image: Image.Image) -> str:
        img_base64 = self._encode_image(image)
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llava",
            "prompt": "Describe this scene in a concise sentence.",
            "images": [img_base64],
            "stream": False
        }
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get("response", "Error: No response from LLaVA.")
        except requests.exceptions.RequestException as e:
            return f"Error: LLaVA API request failed - {e}"

    def describe_video(self):
        if not os.path.exists(self.video_path):
            print(f"Error: Video file not found at {self.video_path}")
            return []
        
        vidcap = cv2.VideoCapture(self.video_path)
        if not vidcap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return []

        descriptions = []
        frame_count = 0
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(fps / self.frame_rate))

        while True:
            success, frame_bgr = vidcap.read()
            if not success:
                break
            if frame_count % interval == 0:
                timestamp_sec = round(frame_count / fps, 2)
                print(f"Analyzing frame at {timestamp_sec:.2f}s...")
                image_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                description = self._generate_description_llava(image_rgb)
                descriptions.append({
                    "timestamp_sec": timestamp_sec,
                    "description": description.strip()
                })
            frame_count += 1
        vidcap.release()
        return descriptions

def create_master_context(transcription, video_descriptions):
    context = "--- START OF VIDEO ANALYSIS CONTEXT ---\n\n"
    
    context += "--- AUDIO TRANSCRIPT ---\n"
    if transcription:
        context += transcription + "\n\n"
    else:
        context += "No audio was transcribed from the video.\n\n"
        
    context += "--- VISUAL EVENTS (Frame by Frame) ---\n"
    if video_descriptions:
        for item in video_descriptions:
            context += f"Time {item['timestamp_sec']:.2f}s: {item['description']}\n"
    else:
        context += "No visual events were described.\n"
        
    context += "\n--- END OF VIDEO ANALYSIS CONTEXT ---\n"
    return context

def ask_gemma_qna(context, question, model_name="gemma"):
    url = "http://localhost:11434/api/generate"
    
    history_text = "\n".join(conversation_history)

    prompt = f"""{context}

    --- RECENT CONVERSATION HISTORY ---
    {history_text}

    --- INSTRUCTION ---
    Based on the video context AND the recent conversation history, answer the user's new question.
    Do not use external knowledge. If the answer is not in the context, say so.

    New Question: {question}
    Answer:"""

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "Error: No response from Gemma.").strip()
    except requests.exceptions.RequestException as e:
        return f"Error: Gemma API request failed - {e}"

if __name__ == "__main__":
    video_path = 'data/cat_biting.mp4'
    gemma_model = 'gemma3:4b'

    print("Starting video analysis. This may take some time...")
    
    audio_extractor = AudioExtractor(video_path)
    transcription = ""
    if audio_extractor.extract():
        transcription = transcribe_audio(audio_extractor.audio_path)
    
    video_describer = VideoDescriber(video_path, frame_rate=0.5)
    video_descriptions = video_describer.describe_video()

    master_context = create_master_context(transcription, video_descriptions)
    
    print("\n\n✅ Analysis complete. You can now ask questions about the video.")
    print("   Type 'quit' or 'exit' to end the session.")
    
    conversation_history = []
    while True:
        user_question = input("\nYou: ")
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("Gemma: Goodbye!")
            break
        
        if not user_question:
            continue
            
        print("Gemma: Thinking...")
        gemma_answer = ask_gemma_qna(master_context, user_question, model_name=gemma_model)
        
        print(f"Gemma: {gemma_answer}")