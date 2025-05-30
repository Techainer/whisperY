import os
import torch
import time
import gradio as gr
from yt_dlp import YoutubeDL
from whisperx.asr import load_model
from whisperx.audio import load_audio

# Default settings
DEFAULT_MODEL_NAME = "large-v3"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_BATCH_SIZE = 1
DEFAULT_CHUNK_SIZE = 30

# Load the model
model = load_model(DEFAULT_MODEL_NAME, device=DEFAULT_DEVICE)

def download_youtube_video(video_url):
    """
    Download a YouTube video and extract its audio file.

    :param video_url: URL of the YouTube video.
    :return: File path of the downloaded video and audio.
    """
    try:
        with YoutubeDL() as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            video_path = ydl.prepare_filename(info_dict)

        return video_path
    except Exception as e:
        return f"Error: {e}"

def transcribe_file(audio_file, batch_size, chunk_size, progress=gr.State()):
    """
    Transcribe the given audio file with user-defined settings.
    """
    print("transcribe_file")
    transcription = ""  # Initialize an empty string to accumulate results
    start_time = time.time()
    
    if audio_file is None or not os.path.exists(audio_file):
        print("Please download file before run transcribe!")
        return "Please download file before run transcribe!", ""
    try:
        # Load the audio file
        audio = load_audio(audio_file)
        # Perform transcription
        for result in model.transcribe(audio, batch_size=batch_size, chunk_size=chunk_size):
            if isinstance(result, str): 
                elapsed_time = time.time() - start_time
                transcription += " " + result.strip()
                transcription = " ".join(transcription.split())
                yield transcription, f"Elapsed time: {elapsed_time:.2f} seconds"
    except Exception as e:
        return f"Error: {e}", "Error occurred"

# Define the Gradio app
with gr.Blocks() as app:
    gr.Markdown("# Techainer AI - Speech to Text")
    gr.Markdown("Upload file (video/audio) or provide a YouTube video URL to get its transcription.")

    with gr.Tab("Transcribe File"):
        with gr.Row():
            with gr.Column():
                upload_file = gr.Audio(label="Upload File (Video/Audio)", type="filepath", recording=True, max_length=20*60)
                batch_size_input = gr.Number(label="Batch Size", value=DEFAULT_BATCH_SIZE, precision=0, visible=False)
                chunk_size_input = gr.Number(label="Chunk Size (seconds)", value=DEFAULT_CHUNK_SIZE, precision=0, visible=False)
                progress_output = gr.Textbox(value="Progress will be displayed here.", interactive=False)
                transcribe_button = gr.Button("Transcribe", variant="primary")

            with gr.Column():
                transcript_output = gr.Textbox(label="Transcription (streaming output)", lines=30, interactive=False, show_copy_button=True)

            transcribe_button.click(
                transcribe_file, 
                inputs=[upload_file, batch_size_input, chunk_size_input], 
                outputs=[transcript_output, progress_output],
            )

    with gr.Tab("Transcribe YouTube Video"):
        with gr.Row():
            with gr.Column():   
                video_url_input = gr.Textbox(label="YouTube Video URL", value="https://www.youtube.com/watch?v=5RSL_9eaHg8")
                video_preview = gr.Video(label="YouTube Video Preview")
                progress_output = gr.Textbox(value="Download progress will be displayed here.", interactive=False)
                with gr.Row():  
                    download_button = gr.Button("Download", variant="primary")
                    transcribe_youtube_button = gr.Button("Transcribe", variant="primary")
                
                def handle_youtube_download(video_url):
                    video_path = download_youtube_video(video_url)
                    if video_path.startswith("Error"):
                        return video_path, None
                    return "Download complete!", video_path
                
                download_button.click(
                    handle_youtube_download,
                    inputs=[video_url_input],
                    outputs=[progress_output, video_preview],
                )

            with gr.Column():
                batch_size_input_youtube = gr.Number(label="Batch Size", value=DEFAULT_BATCH_SIZE, precision=0, visible=False)
                chunk_size_input_youtube = gr.Number(label="Chunk Size (seconds)", value=DEFAULT_CHUNK_SIZE, precision=0, visible=False)
                transcript_output_youtube = gr.Textbox(label="Transcription (streaming output)", lines=30, interactive=False, show_copy_button=True)

            transcribe_youtube_button.click(
                transcribe_file,
                inputs=[video_preview, batch_size_input_youtube, chunk_size_input_youtube],
                outputs=[transcript_output_youtube, progress_output],
            )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", max_file_size="50mb")
