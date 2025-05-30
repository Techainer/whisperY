<h1 align="center">WhisperY: Optimized version of whisper v3 model for Vietnamese Speech Recognition </h1>

WhisperY base on [WhisperX](https://github.com/m-bain/whisperX) repo.

### 1. Installation

```bash
conda create --name whispery python=3.10
conda activate whispery
pip install -r requirements.txt
```

You may also need to install ffmpeg, rust etc. Follow openAI instructions here https://github.com/openai/whisper#setup.

### 2. Run Gradio demo
```bash
python app.py

# Gradio app: https://localhost:7860
```