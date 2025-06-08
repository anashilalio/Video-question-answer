from moviepy.editor import VideoFileClip
import os
import whisper

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
                clip.audio.write_audiofile(self.audio_path, codec='pcm_s16le', fps=16000)
                print(f"Audio extracted to {self.audio_path}")
                return True
        except Exception as e:
            print(f"Error during audio extraction: {e}")
            return False

def transcribe_audio(audio_path, model_name="base"):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    print("Transcribing audio...")
    result = model.transcribe(audio_path)
    return result.get("text", "Transcription failed.")

if __name__ == "__main__":
    video_path = 'data/LLM.mp4'
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
    else:
        extractor = AudioExtractor(video_path)
        if extractor.extract():
            try:
                transcription = transcribe_audio(extractor.audio_path)
                print("\n--- Transcription ---")
                print(transcription)
            except Exception as e:
                print(f"An error occurred during transcription: {e}")