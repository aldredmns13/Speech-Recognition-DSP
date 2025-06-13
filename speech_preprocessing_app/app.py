import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import noisereduce as nr
from scipy.signal import butter, lfilter
from io import BytesIO

# --- DSP Functions ---
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
    fig, ax = plt.subplots(figsize=(5, 2))
    t = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(t, audio, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

# --- AudioProcessor Class ---
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32000.0
        self.frames.append(audio)
        return frame

# --- Streamlit Page Settings ---
st.set_page_config(page_title="Speech Preprocessing App", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎤 Speech Recognition Preprocessing</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>“Clean your noisy audio with us.”</h3>", unsafe_allow_html=True)
st.divider()

# --- Input Choice Section ---
st.subheader("📌 Choose Your Input Method")
col1, col2 = st.columns(2)

sr = 48000
audio_data = None

with col1:
    mic_record = st.button("🎙 Speak Using Microphone")

with col2:
    uploaded_file = st.file_uploader("📁 Upload Audio File", type=["wav"])

# --- Microphone Recording ---
if mic_record:
    st.info("Click 'Start' below to record. Speak for 5–10 seconds, then click '✅ Process'.")
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
                audio_data = raw_audio[-sr * 10:]  # Last 10 seconds
        else:
            st.warning("No audio detected.")

# --- File Upload Handling ---
elif uploaded_file:
    y, _ = librosa.load(uploaded_file, sr=sr, mono=True)
    audio_data = y

# --- Processing and Display ---
if audio_data is not None:
    st.markdown("## 🔊 Your Audio")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🎧 Original Audio**")
        buf_orig = BytesIO()
        sf.write(buf_orig, audio_data, sr, format='wav')
        st.audio(buf_orig, format='audio/wav')
        plot_waveform(audio_data, sr, "Original")

    with col2:
        st.markdown("🛠️ **Cleaning Your Audio...**")
        clean_audio = normalize_audio(audio_data)
        clean_audio = reduce_noise(clean_audio, sr)
        clean_audio = bandpass_filter(clean_audio, sr)
        clean_audio = amplify_audio(clean_audio)
        clean_audio = normalize_audio(clean_audio)

        buf_clean = BytesIO()
        sf.write(buf_clean, clean_audio, sr, format='wav')

        st.success("✅ Cleaning Complete!")

    st.divider()
    st.markdown("## 🔁 Know the Difference")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🪞 **Old Audio**")
        st.audio(buf_orig, format='audio/wav')
        plot_waveform(audio_data, sr, "Before Cleaning")

    with col2:
        st.markdown("✨ **Updated Audio**")
        st.audio(buf_clean, format='audio/wav')
        plot_waveform(clean_audio, sr, "After Cleaning")

    st.download_button("⬇️ Download Cleaned Audio", buf_clean.getvalue(), "cleaned_audio.wav", mime="audio/wav")

    
