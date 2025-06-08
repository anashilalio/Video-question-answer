# WARNING: This approach of processing a video second-by-second is extremely
# slow and resource-intensive. Transcribing audio in one-second chunks with
# Whisper is highly inefficient and can produce low-quality results.
# A better approach is to transcribe the entire audio file at once to get
# timestamps and then align those with frame descriptions.

import os
# This patch may be needed by older libraries for Python 3 compatibility.
import inspect
from collections import namedtuple
if not hasattr(inspect, 'ArgSpec'):
    inspect.ArgSpec = namedtuple('ArgSpec', 'args varargs keywords defaults')

import whisper
import cv2
import base64
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from moviepy.editor import VideoFileClip
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class AdvancedVideoDescriber:
    def __init__(self, video_path, output_dir="output"):
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        print("Loading models... (This might take a while)")
        self.whisper_model = whisper.load_model("tiny")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Models loaded.")

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """Encodes a PIL Image to a base64 string."""
        buf = BytesIO()
        image.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()

    def _generate_image_description(self, image: Image.Image, prompt="Describe this image concisely.") -> str:
        """Generates a description for an image using a local LLaVA model."""
        img_b64 = self._encode_image(image)
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llava", "prompt": prompt, "images": [img_b64], "stream": False}
            )
            resp.raise_for_status()
            return resp.json().get("response", "No description returned.")
        except requests.exceptions.RequestException as e:
            return f"API Error: {e}"

    def _transcribe_audio_chunk(self, audio_array: np.ndarray, start_sec: int, sample_rate: int) -> str:
        """Transcribes a one-second chunk of an audio array."""
        start_sample = start_sec * sample_rate
        chunk = audio_array[start_sample : start_sample + sample_rate]
        
        padded_chunk = whisper.pad_or_trim(chunk)
        mel = whisper.log_mel_spectrogram(padded_chunk).to(self.whisper_model.device)
        options = whisper.DecodingOptions(fp16=False, without_timestamps=True)
        result = self.whisper_model.decode(mel, options)
        return result.text.strip()

    def process_video(self):
        """Processes a video to generate and compare image and audio descriptions per second."""
        if not os.path.exists(self.video_path):
            print(f"Error: Video file not found: {self.video_path}")
            return []

        print("Step 1: Extracting audio...")
        try:
            with VideoFileClip(self.video_path) as clip:
                audio_path = os.path.join(self.output_dir, "temp_audio.wav")
                clip.audio.write_audiofile(audio_path, fps=16000, codec='pcm_s16le')
                total_sec = int(clip.duration)
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return []

        print("Step 2: Loading audio into memory...")
        audio_array = whisper.load_audio(audio_path)
        
        print(f"Step 3: Processing video for {total_sec} seconds...")
        cap = cv2.VideoCapture(self.video_path)
        results = []

        for sec in range(total_sec):
            print(f"--- Processing Second: {sec} ---")
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            success, frame = cap.read()
            if not success: continue

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            text_image = self._generate_image_description(image)
            text_audio = self._transcribe_audio_chunk(audio_array, sec, whisper.audio.SAMPLE_RATE)

            if text_image and text_audio:
                embeddings = self.embed_model.encode([text_image, text_audio])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            else:
                similarity = 0.0

            results.append({
                "second": sec,
                "image_description": text_image,
                "audio_transcription": text_audio,
                "similarity": round(float(similarity), 4)
            })

        cap.release()
        return results

if __name__ == "__main__":
    video_file_path = "data/LLM.mp4"
    if not os.path.exists(video_file_path):
        print(f"Video file not found: {video_file_path}")
    else:
        describer = AdvancedVideoDescriber(video_path=video_file_path)
        outputs = describer.process_video()

        print("\n\n--- FINAL RESULTS ---")
        for item in outputs:
            print(f"Second {item['second']} — Similarity: {item['similarity']}")
            print(f"  Image Desc: {item['image_description']}")
            print(f"  Audio Text: {item['audio_transcription']}")
            print("-" * 80)