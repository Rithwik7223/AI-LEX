import gradio as gr
import torch
from TTS.api import TTS
from scipy.io.wavfile import write
import os

# --- Configuration ---
OUTPUT_DIR = "cloned_voices"
SPEAKER_WAV = os.path.join(OUTPUT_DIR, "speaker.wav")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load the AI Model ---
# This will download the model on the first run
print("Loading XTTS model, this may take a moment...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print("Model loaded successfully.")

# --- Core Functions ---
def clone_voice(audio_input):
    """Saves the recorded audio as the speaker reference."""
    if audio_input is None:
        return "Recording failed. Please try again.", None

    sample_rate, audio_data = audio_input
    
    # Save the recorded audio to a file
    write(SPEAKER_WAV, sample_rate, audio_data)
    
    status_message = f"Voice successfully cloned and saved as speaker.wav"
    print(status_message)
    return status_message, gr.Audio(value=SPEAKER_WAV)

def chat(text_input):
    """Generates speech from text using the cloned voice."""
    if not os.path.exists(SPEAKER_WAV):
        return None, "Error: No voice has been cloned yet. Please record your voice first."

    if not text_input:
        return None, "Please enter some text to generate speech."

    output_file = os.path.join(OUTPUT_DIR, "response.wav")

    print(f"Generating speech for text: '{text_input}'")
    # Generate speech using the cloned voice
    tts.tts_to_file(
        text=text_input,
        speaker_wav=SPEAKER_WAV,
        language="en", # Change language code if needed (e.g., "es", "fr")
        file_path=output_file
    )
    print(f"Speech generated and saved to {output_file}")
    
    return gr.Audio(value=output_file), f"You said: {text_input}"

# --- Gradio User Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Modern EchoBot v2: Powered by Coqui XTTS")
    gr.Markdown("A sophisticated, high-quality voice cloning chatbot.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1. Clone Your Voice")
            gr.Markdown("Record 5-10 seconds of clear speech. For best results, speak naturally.")
            mic_input = gr.Audio(sources=["microphone"], type="numpy", label="Your Voice Input")
            clone_button = gr.Button("Clone My Voice", variant="primary")
        
        with gr.Column(scale=2):
            gr.Markdown("## 2. Review Your Voice Sample")
            gr.Markdown("Listen to the audio you recorded to ensure it's clear before chatting.")
            cloned_audio_output = gr.Audio(label="Your Cloned Voice Sample")
            status_text = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 3. Chat with Your Cloned Voice")
            text_input = gr.Textbox(label="Enter your message", lines=3)
            chat_button = gr.Button("Generate Speech", variant="primary")
            audio_output = gr.Audio(label="Bot's Audio Response")
            text_output = gr.Textbox(label="Bot's Text Response", interactive=False)

    # --- Connect UI components to functions ---
    clone_button.click(
        fn=clone_voice, 
        inputs=mic_input, 
        outputs=[status_text, cloned_audio_output]
    )
    chat_button.click(
        fn=chat, 
        inputs=text_input, 
        outputs=[audio_output, text_output]
    )

print("Launching Gradio Interface...")
demo.launch()