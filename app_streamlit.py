import streamlit as st
import torch
import time
import tempfile
from whisperx.asr import load_model
from whisperx.audio import load_audio

# Default settings
DEFAULT_MODEL_NAME = "large-v3"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_BATCH_SIZE = 1
DEFAULT_CHUNK_SIZE = 30

# Initialize the ASR model
@st.cache_resource
def load_asr_model():
    return load_model(DEFAULT_MODEL_NAME, device=DEFAULT_DEVICE)

model = load_asr_model()

def transcribe_audio(audio_file_path, batch_size, chunk_size):
    """
    Transcribe the given audio file with user-defined settings.
    """
    transcription = ""
    start_time = time.time()
    try:
        # Load the audio file
        audio = load_audio(audio_file_path)
        
        # Perform transcription
        for result in model.transcribe(audio, batch_size=batch_size, chunk_size=chunk_size):
            if isinstance(result, str):
                elapsed_time = time.time() - start_time
                transcription += result + "\n"  # Append each result with a newline
                # Update progress and transcription
                st.session_state.transcription = transcription
                st.session_state.progress = f"Elapsed time: {elapsed_time:.2f} seconds"
                time.sleep(0.1)  # Simulate streaming
    except Exception as e:
        st.session_state.transcription = f"Error: {e}"
        st.session_state.progress = "Error occurred"

# Initialize session state
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "progress" not in st.session_state:
    st.session_state.progress = "Progress will be displayed here."

# Inject custom CSS to remove padding
st.markdown(
    """
    <style>
        .css-18e3th9 {padding: 0 !important;}  /* Remove padding from main container */
        .css-1d391kg {padding: 0 !important;}  /* Remove padding from top container */
        .css-1avcm0n {padding: 0 !important;}  /* Remove padding from column container */
    </style>
    """,
    unsafe_allow_html=True,
)

# Full-width container
with st.container():
    st.title("Audio Transcription App")
    st.markdown("Upload an audio file, adjust settings, and transcribe it in real-time.")

    # Two-column layout without padding
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a"])
        batch_size = st.number_input("Batch Size", value=DEFAULT_BATCH_SIZE, step=1, min_value=1)
        chunk_size = st.number_input("Chunk Size (seconds)", value=DEFAULT_CHUNK_SIZE, step=1, min_value=1)
        if st.button("Transcribe"):
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name
                with st.spinner("Transcription in progress..."):
                    transcribe_audio(temp_file_path, batch_size, chunk_size)
            else:
                st.warning("Please upload an audio file before transcribing.")

    with col2:
        st.header("Output")
        st.text_area("Progress", value=st.session_state.progress, disabled=True)
        st.text_area("Transcription", value=st.session_state.transcription, height=400, disabled=True)
