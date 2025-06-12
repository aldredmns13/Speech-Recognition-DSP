import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import noisereduce as nr
from scipy.signal import butter, lfilter
from io import BytesIO

# --- DSP Helper Functions ---

def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    return audio if max_val == 0 else audio / max_val * 0.9

def reduce_noise(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr)

def bandpass_filter(audio, sr=48000, lowcut=500.0, highcut=3000.0):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(6, [low, high], btype='band')
    return lfilter(b, a, audio)

def amplify_audio(audio, gain=2.0):
    return np.clip(audio * gain, -1.0, 1.0)

def plot_waveform(audio, sr, title):
    fig, ax = plt.subplots()
    t = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(t, audio)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

# --- AudioProcessor for Microphone ---

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32000.0  # Convert int16 to float32
        self.frames.append(audio)
        return frame

# --- Streamlit UI ---

st.title("🎤 Speech Preprocessing App (Mic & File Upload)")

sr = 48000
input_method = st.radio("Choose input method:", ["🎙 Record via Microphone", "📁 Upload WAV File"])

# --- Process Function (Shared) ---

def process_and_display(audio, sr):
    st.subheader("🔊 Original Audio")
    buf_orig = BytesIO()
    sf.write(buf_orig, audio, sr, format='wav')
    st.audio(buf_orig)
    plot_waveform(audio, sr, "Original Audio")

    # Apply processing steps
    audio = normalize_audio(audio)
    audio = reduce_noise(audio, sr)
    audio = bandpass_filter(audio, sr)
    audio = amplify_audio(audio)
    audio = normalize_audio(audio)

    st.subheader("🧼 Cleaned Audio")
    buf_clean = BytesIO()
    sf.write(buf_clean, audio, sr, format='wav')
    st.audio(buf_clean)
    plot_waveform(audio, sr, "Cleaned Audio")

    st.download_button("⬇️ Download Cleaned Audio", buf_clean.getvalue(), "cleaned_audio.wav", mime="audio/wav")

# --- File Upload Path ---

if input_method == "📁 Upload WAV File":
    uploaded = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded:
        y, file_sr = librosa.load(uploaded, sr=sr, mono=True)
        process_and_display(y, sr)

# --- Microphone Path ---

elif input_method == "🎙 Record via Microphone":
    st.info("Click start and speak for 5–10 seconds. Then press 'Process'.")

    webrtc_ctx = webrtc_streamer(
        key="mic",
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    if st.button("✅ Process Mic Recording"):
        if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.audio_processor:
            raw_audio = np.concatenate(webrtc_ctx.audio_processor.frames)
            if len(raw_audio) < sr * 2:
                st.warning("Please record at least 2 seconds.")
            else:
                audio = raw_audio[-sr * 10:]  # last 10 seconds max
                process_and_display(audio, sr)
        else:
            st.warning("No audio data available.")
