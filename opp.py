
import streamlit as st
import os
import requests
from moviepy.editor import VideoFileClip
import whisper
import cv2
import base64
from io import BytesIO
from PIL import Image
import tempfile
import json
from datetime import datetime
import re

st.set_page_config(layout="centered", page_title="Video QA Bot")


st.markdown(
    """
    <style>
    :root {
        --background-color: #0E1117;
        --text-color: #FAFAFA;
        --user-color: #262730;
        --assistant-color: #1E1F26;
    }
    
    body, .stApp { 
        background-color: var(--background-color); 
        color: var(--text-color); 
    }
    #MainMenu, footer, header { 
        visibility: hidden; 
    }
    
    /* --- Main Chat Interface Styles --- */

    /* This is the new, user-provided rule to hide the assistant's icon. */
    /* It is guaranteed to work because you found the name yourself. */
    .st-emotion-cache-jmw8un {
        display: none !important;
    }
    .st-emotion-cache-janbn0 {
        margin-left: 50% !important;
    }

    /* We also hide the user's icon container just in case it's different */
    div[data-testid="stChatMessageAvatarUser"] {
        display: none !important;
    }

    /* Styles the container for the user's message to push it right */
    div[data-testid="stChatMessage"]:has(div[aria-label="user message"]) {
        display: flex;
        justify-content: flex-end;
        margin-right:50%
    }
    
    /* Styles the user's message bubble */
    div[aria-label="user message"] > div {
        background-color: var(--user-color);
        padding: 1rem;
        border-radius: 1rem;
        max-width: 70%;
    }

    /* Styles the assistant's message bubble to be transparent */
    div[aria-label="assistant message"] > div {
        background-color: transparent;
        padding: 0;
    }
    .setVideo{
        height : 100px;
        width: 100px;
    }
    video {
    /* Set a fixed height for the video player. You can change this value. */
    height: 450px !important;

    /* This makes the video fill the entire frame, which often looks cleaner */
    object-fit: cover !important;
    
    /* Optional: Add rounded corners to match the rest of your UI */
    border-radius: 0.7rem; 
}
    </style>
    """,
    unsafe_allow_html=True,
)

class AudioExtractor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.audio_path = os.path.splitext(video_path)[0] + '.wav'
    def extract(self):
        if not os.path.exists(self.video_path): return False
        try:
            with VideoFileClip(self.video_path) as clip:
                if clip.audio is None: return False
                clip.audio.write_audiofile(self.audio_path, codec='pcm_s16le', fps=16000, logger=None)
                return True
        except Exception: return False

def transcribe_audio(audio_path, model_name="base"):
    if not os.path.exists(audio_path): return "No audio file found."
    model = whisper.load_model(model_name)
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
        payload = { "model": "llava", "prompt": "Describe this scene concisely.", "images": [img_base64], "stream": False }
        try:
            response = requests.post(url, json=payload, timeout=90)
            response.raise_for_status()
            return response.json().get("response", "LLaVA Error")
        except requests.exceptions.RequestException: return "LLaVA API request failed"
    def describe_video(self):
        if not os.path.exists(self.video_path): return []
        vidcap = cv2.VideoCapture(self.video_path)
        if not vidcap.isOpened(): return []
        descriptions = []
        frame_count = 0
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(fps / self.frame_rate)) if fps > 0 else 1
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.sidebar.progress(0, text="Analyzing frames...")
        while True:
            success, frame_bgr = vidcap.read()
            if not success: break
            if frame_count % interval == 0:
                timestamp_sec = round(frame_count / fps, 2) if fps > 0 else 0
                image_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                description = self._generate_description_llava(image_rgb)
                descriptions.append({ "timestamp_sec": timestamp_sec, "description": description.strip() })
                progress = min(1.0, (frame_count + 1) / total_frames) if total_frames > 0 else 1.0
                progress_bar.progress(progress, text=f"Analyzing frame at {timestamp_sec:.2f}s...")
            frame_count += 1
        progress_bar.progress(1.0, text="Frame analysis complete.")
        vidcap.release()
        return descriptions

def create_master_context(transcription, video_descriptions):
    context = "--- VIDEO ANALYSIS CONTEXT ---\n"
    context += "AUDIO TRANSCRIPT: " + (transcription if transcription else "N/A") + "\n\n"
    context += "VISUAL EVENTS:\n" + ("\n".join([f"Time {item['timestamp_sec']:.2f}s: {item['description']}" for item in video_descriptions]) if video_descriptions else "N/A")
    context += "\n--- END OF CONTEXT ---\n"
    return context


def get_query_category(question, model_name="gemma3:4b"):
    """First, ask the AI to categorize the user's question."""
    url = "http://localhost:11434/api/generate"
    
    prompt = f"""You are a query routing assistant. Your job is to determine what kind of information the user is asking for.
Based on the user's question, should I search in the 'AUDIO_TRANSCRIPT', the 'VISUAL_EVENTS', or 'GENERAL_SUMMARY'?
- If the question is about what someone said, mentioned, or talked about, respond with only the word: AUDIO_TRANSCRIPT
- If the question is about what happened at a specific time, or what something looked like, respond with only the word: VISUAL_EVENTS
- For all other general questions, respond with only the word: GENERAL_SUMMARY

User's question: "{question}"
CATEGORY:"""

    payload = { "model": model_name, "prompt": prompt, "stream": False, "options": {"temperature": 0.0} }
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        category = response.json().get("response", "GENERAL_SUMMARY").strip()
        if category not in ["AUDIO_TRANSCRIPT", "VISUAL_EVENTS", "GENERAL_SUMMARY"]:
            return "GENERAL_SUMMARY"
        return category
    except requests.exceptions.RequestException:
        return "GENERAL_SUMMARY"



def ask_gemma_qna(context, question, model_name="gemma3:4b"):
    """The definitive two-step Q&A function with highly specific prompts."""
    
    category = get_query_category(question, model_name)
    print(f"Query Category: {category}") 

    try:
        audio_transcript = context.split("AUDIO TRANSCRIPT:")[1].split("VISUAL EVENTS:")[0].strip()
        visual_events = context.split("VISUAL EVENTS:")[1].split("--- END OF CONTEXT ---")[0].strip()
    except IndexError:
        audio_transcript = "Not available."
        visual_events = "Not available."

    instruction = ""
    focused_context = ""

    if category == "AUDIO_TRANSCRIPT":
        focused_context = f"--- START OF AUDIO TRANSCRIPT ---\n{audio_transcript}\n--- END OF AUDIO TRANSCRIPT ---"
        instruction = "The user is asking what was said. Directly quote or summarize the content from the AUDIO TRANSCRIPT. Begin your answer with 'According to the transcript...' or 'He said...'"
    
    elif category == "VISUAL_EVENTS":
        focused_context = f"--- START OF VISUAL EVENTS ---\n{visual_events}\n--- END OF VISUAL EVENTS ---"
        instruction = "Based ONLY on the timestamped visual events provided, answer the user's question about what is happening."
    
    else: 
        focused_context = context
        instruction = "Based on the full context, answer the following question:"

    url = "http://localhost:11434/api/generate"
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.get('conversation_history', [])])
    
    final_prompt = f"""You are a Video QA Assistant. Follow the user's instruction precisely.
--- RELEVANT CONTEXT ---
{focused_context}

--- CONVERSATION HISTORY ---
{history_text}

--- INSTRUCTION ---
{instruction}
If the information is not available in the provided context, state that clearly.
Question: {question}
Answer:"""
    
    payload = { "model": model_name, "prompt": final_prompt, "stream": False }
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "Gemma Error").strip()
    except requests.exceptions.RequestException:
        st.error("Connection to Ollama failed. Is it running?")
        return None
def get_chat_title(context, model_name="gemma3:4b"):
    url = "http://localhost:11434/api/generate"
    prompt = f"""Based on the following video analysis context, create a very short, descriptive title for this video in 4-6 words. This title will be used as a filename. Do not use special characters. Example: Elon Musk giving a speech at an event.\n\nCONTEXT:\n{context}\n\nTITLE:"""
    payload = { "model": model_name, "prompt": prompt, "stream": False }
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "Untitled Chat").strip()
    except requests.exceptions.RequestException:
        return "Untitled Chat"

def slugify(text):
    text = text.lower()
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'[^\w\-]', '', text)
    return text

HISTORY_DIR = "chat_history"
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

def save_chat(filename, video_path, master_context, history):
    filepath = os.path.join(HISTORY_DIR, filename)
    chat_data = {
        "original_video_path": video_path, "master_context": master_context, "conversation_history": history
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=4, ensure_ascii=False)

def load_chat(filename):
    filepath = os.path.join(HISTORY_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        chat_data = json.load(f)
    st.session_state.active_chat_id = filename
    st.session_state.video_path = chat_data["original_video_path"]
    st.session_state.master_context = chat_data["master_context"]
    st.session_state.conversation_history = chat_data["conversation_history"]
    st.session_state.analysis_complete = True
    st.session_state.new_chat_started = False



def get_saved_chats():
    """Gets a list of all saved chat JSON files, sorted by modification time (newest first)."""
    try:
        files = [os.path.join(HISTORY_DIR, f) for f in os.listdir(HISTORY_DIR) if f.endswith('.json')]
        
        
        files.sort(key=os.path.getmtime, reverse=True)
        
        return [os.path.basename(f) for f in files]
    except FileNotFoundError:
        return []

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "new_chat_started" not in st.session_state:
    st.session_state.new_chat_started = True

with st.sidebar:
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.active_chat_id = None
        st.session_state.conversation_history = []
        st.session_state.new_chat_started = True
        st.session_state.video_path = None
        st.session_state.analysis_complete = False
        st.rerun()

    st.markdown("---")
    
    if st.session_state.new_chat_started:
        st.header("Start New Chat")
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
        
        if uploaded_file:
            if st.button("Analyze Video", type="primary", use_container_width=True):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmpfile:
                    tmpfile.write(uploaded_file.getvalue())
                    st.session_state.video_path = tmpfile.name
                
                with st.status("Performing full video analysis...", expanded=True) as status:
                    status.write("Extracting audio...")
                    audio_extractor = AudioExtractor(st.session_state.video_path)
                    transcription = ""
                    if audio_extractor.extract():
                        status.write("Transcribing audio...")
                        transcription = transcribe_audio(audio_extractor.audio_path)
                    
                    status.write("Analyzing video frames...")
                    video_describer = VideoDescriber(st.session_state.video_path)
                    video_descriptions = video_describer.describe_video()
                    
                    status.write("Creating final context...")
                    st.session_state.master_context = create_master_context(transcription, video_descriptions)
                    
                    status.write("Generating a title for the chat...")
                    chat_title = get_chat_title(st.session_state.master_context, model_name='gemma3:4b')
                    slug_title = slugify(chat_title)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_chat_filename = f"{slug_title}_{timestamp}.json"

                    st.session_state.analysis_complete = True
                    st.session_state.new_chat_started = False
                    st.session_state.conversation_history = [{"role": "assistant", "content": "Analysis complete. How can I help?"}]
                    
                    st.session_state.active_chat_id = new_chat_filename
                    save_chat(new_chat_filename, st.session_state.video_path, st.session_state.master_context, st.session_state.conversation_history)
                    status.update(label="✅ Analysis Complete!", state="complete")
                st.rerun()

    st.markdown("---")
    st.header("Chat History")
    for chat_file in get_saved_chats():
        st.markdown(f'<div class="history-item">', unsafe_allow_html=True)
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(chat_file, key=chat_file, use_container_width=True):
                load_chat(chat_file)
                st.rerun()
        with col2:
            st.markdown('<div class="delete-button">', unsafe_allow_html=True)
            if st.button("🗑️", key=f"delete_{chat_file}", use_container_width=True, help=f"Delete chat {chat_file}"):
                os.remove(os.path.join(HISTORY_DIR, chat_file))
                if st.session_state.get("active_chat_id") == chat_file:
                    st.session_state.active_chat_id = None
                    st.session_state.new_chat_started = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if not st.session_state.get("active_chat_id"):
    st.markdown("<h1 style='text-align: center; color: #4A4B52;'>What can I help with?</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #4A4B52;'>Start a 'New Chat' or load a past conversation from the sidebar.</p>", unsafe_allow_html=True)
else:
    st.video(st.session_state.video_path)
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_gemma_qna(st.session_state.master_context, prompt, model_name='gemma3:4b')
                if response:
                    st.markdown(response)
                    st.session_state.conversation_history.append({"role": "assistant", "content": response})
                    save_chat(st.session_state.active_chat_id, st.session_state.video_path, st.session_state.master_context, st.session_state.conversation_history)
        st.rerun()