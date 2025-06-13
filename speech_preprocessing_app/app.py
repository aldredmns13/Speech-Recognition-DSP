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

def bandpass_filter(audio, sr=16000, lowcut=500.0, highcut=2800.0):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(6, [low, high], btype='band')
    return lfilter(b, a, audio)

def amplify_audio(audio, gain=2.0):
    return np.clip(audio * gain, -1.0, 1.0)

def plot_waveform(audio, sr, title):
    fig, ax = plt.subplots(figsize=(6, 2))
    t = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(t, audio, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

# --- AudioProcessor for Microphone ---

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32000.0
        self.frames.append(audio)
        return frame

# --- Streamlit UI ---

st.set_page_config(page_title="Speech Preprocessing App", layout="wide")
st.title("🎤 Speech Preprocessing App")
st.caption("Record or upload speech and enhance it with noise reduction, filtering, and normalization.")

sr = 48000
input_method = st.radio("🎚 Select Input Method:", ["🎙 Microphone", "📁 Upload WAV File"], horizontal=True)

# --- Process Function ---

def process_and_display(audio, sr):
    with st.expander("🎧 Original Audio and Waveform", expanded=True):
        buf_orig = BytesIO()
        sf.write(buf_orig, audio, sr, format='wav')
        st.audio(buf_orig, format='audio/wav')
        plot_waveform(audio, sr, "Original Audio")

    # Processing pipeline
    audio = normalize_audio(audio)
    audio = reduce_noise(audio, sr)
    audio = bandpass_filter(audio, sr)
    audio = amplify_audio(audio)
    audio = normalize_audio(audio)

    with st.expander("✨ Cleaned Audio and Waveform", expanded=True):
        buf_clean = BytesIO()
        sf.write(buf_clean, audio, sr, format='wav')
        st.audio(buf_clean, format='audio/wav')
        plot_waveform(audio, sr, "Cleaned Audio")

    st.download_button("⬇️ Download Cleaned Audio", buf_clean.getvalue(), "cleaned_audio.wav", mime="audio/wav")

# --- File Upload Path ---

if input_method == "📁 Upload WAV File":
    with st.container():
        st.subheader("📤 Upload Your WAV File")
        uploaded = st.file_uploader("Choose a WAV file (Mono preferred)", type=["wav"])
        if uploaded:
            y, file_sr = librosa.load(uploaded, sr=sr, mono=True)
            st.success("✅ File uploaded successfully.")
            if st.button("🔁 Process Uploaded Audio"):
                process_and_display(y, sr)

# --- Microphone Path ---

elif input_method == "🎙 Microphone":
    st.subheader("🎙 Record via Microphone")
    st.markdown("Click `Start` below to begin recording. After recording for a few seconds, click `✅ Process Mic Recording`.")

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
                st.warning("⚠️ Please record at least 2 seconds of audio.")
            else:
                audio = raw_audio[-sr * 10:]  # Use last 10 seconds max
                process_and_display(audio, sr)
        else:
            st.warning("⚠️ No audio data available. Please start recording first.")
