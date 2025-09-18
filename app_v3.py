import gradio as gr
import torch
from TTS.api import TTS
from scipy.io.wavfile import write
import os
import google.generativeai as genai

# --- Configuration ---
OUTPUT_DIR = "cloned_voices"
SPEAKER_WAV = os.path.join(OUTPUT_DIR, "speaker.wav")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Securely load API Key ---
try:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    print("="*50)
    print("ERROR: GOOGLE_API_KEY environment variable not set.")
    print("Please set the key before running: set GOOGLE_API_KEY=YOUR_API_KEY")
    print("="*50)
    exit()

# --- Load the AI Models ---
print("Loading conversation model (Gemini)...")
llm = genai.GenerativeModel('gemini-1.5-flash')
chat_session = llm.start_chat(history=[]) # Initialize chat history
print("Conversation model loaded.")

print("Loading XTTS voice cloning model, this may take a moment...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print("Voice cloning model loaded successfully.")


# --- Core Functions ---
def clone_voice(audio_input):
    """Saves the recorded audio as the speaker reference."""
    if audio_input is None:
        return "Recording failed. Please try again.", None

    sample_rate, audio_data = audio_input
    write(SPEAKER_WAV, sample_rate, audio_data)
    
    status_message = f"Voice successfully cloned and saved as speaker.wav"
    print(status_message)
    return status_message, gr.Audio(value=SPEAKER_WAV)

def chat(user_message):
    """
    1. Gets a text response from the LLM.
    2. Converts that text response to speech using the cloned voice.
    """
    if not os.path.exists(SPEAKER_WAV):
        return None, "Error: No voice has been cloned yet. Please record your voice first."

    if not user_message:
        return None, "Please enter a message."

    # 1. Get text response from Gemini
    print(f"User message: '{user_message}'")
    response = chat_session.send_message(user_message)
    llm_response_text = response.text
    print(f"LLM Response: '{llm_response_text}'")

    # 2. Convert LLM response to speech
    output_file = os.path.join(OUTPUT_DIR, "response.wav")
    tts.tts_to_file(
        text=llm_response_text,
        speaker_wav=SPEAKER_WAV,
        language="en",
        file_path=output_file
    )
    print(f"Speech generated and saved to {output_file}")
    
    return gr.Audio(value=output_file), llm_response_text

# --- Gradio User Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Conversational EchoBot v3: Powered by Gemini & Coqui XTTS")
    gr.Markdown("An interactive AI chatbot that learns your voice and talks back.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1. Clone Your Voice")
            mic_input = gr.Audio(sources=["microphone"], type="numpy", label="Record 5-10 seconds of clear speech")
            clone_button = gr.Button("Clone My Voice", variant="primary")
        
        with gr.Column(scale=2):
            gr.Markdown("## 2. Review Your Voice Sample")
            cloned_audio_output = gr.Audio(label="Your Cloned Voice Sample")
            status_text = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 3. Chat with Your AI")
            text_input = gr.Textbox(label="Your message", lines=3)
            chat_button = gr.Button("Send Message", variant="primary")
            audio_output = gr.Audio(label="AI's Audio Response")
            text_output = gr.Textbox(label="AI's Text Response", interactive=False)

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