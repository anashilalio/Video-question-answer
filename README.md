# Multi-Modal Video QA Assistant

An interactive web application built with Streamlit and Python that provides a conversational interface for analyzing video content. The system uses a multi-modal AI pipeline to "watch" video frames, "listen" to audio, and "reason" about the combined information to answer user questions in real-time.
![Untitled video - Made with Clipchamp (1)](https://github.com/user-attachments/assets/2199ce18-38df-4682-a11c-4fac858390cd)

video used in this example https://www.youtube.com/shorts/0BDttBaX4A4

## Features

- **Multi-modal Analysis**: Combines visual frame analysis and audio transcription
- **Real-time Interaction**: Ask questions about video content and get intelligent responses
- **Streamlit Interface**: User-friendly web application for easy video upload and interaction
- **Modular Architecture**: Specialized AI models working together through a Python backend

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Computer Vision**: LLaVA (Large Language and Vision Assistant)
- **Language Model**: Gemma3
- **Audio Processing**: OpenAI Whisper
- **Video Processing**: MoviePy, OpenCV

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/anashilalio/Video-question-answer
cd Video_QA
```

### 2. Create Virtual Environment

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install AI Models

Install the required Ollama models:

```bash
ollama run llava
ollama run gemma3:4b ##choose the version of gemma3 you want 4 , 12 ...
```

### 4. Install Python Dependencies


```text
streamlit
requests
moviepy
openai-whisper
opencv-python-headless
Pillow
# Specific torch version to avoid Streamlit compatibility issues
torch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Start Ollama**:
   

2. **Run the Streamlit application**:
   ```bash
   streamlit run opp.py
   ```

3. **Open your browser** and navigate to `http://localhost:8501`

4. **Upload a video file** using the interface

5. **Ask questions** about the video content and receive AI-powered responses

## Architecture

The system follows a modular architecture where:

- **Video Processing**: Extracts frames and audio from uploaded videos
- **Visual Analysis**: LLaVA model processes video frames for visual understanding
- **Audio Analysis**: Whisper transcribes and analyzes audio content
- **Response Generation**: Gemma model synthesizes multi-modal information to generate responses
- **Web Interface**: Streamlit provides an intuitive user interface



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- [Ollama](https://ollama.ai/) for providing local AI model infrastructure
- [LLaVA](https://llava-vl.github.io/) for multi-modal vision-language understanding
- [OpenAI Whisper](https://openai.com/research/whisper) for speech recognition
- [Streamlit](https://streamlit.io/) for the web application framework
