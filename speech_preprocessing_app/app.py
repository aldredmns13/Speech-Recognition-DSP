import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.signal import firwin, lfilter
import noisereduce as nr

# ------------------ DSP Functions ------------------

def apply_fir_bandpass(audio, sr, lowcut=300.0, highcut=3400.0, numtaps=101):
    fir_coeff = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=sr)
    return lfilter(fir_coeff, 1.0, audio)

def reduce_noise(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr)

def normalize_audio(audio):
    return audio / np.max(np.abs(audio)) * 0.9

def plot_waveform(audio, sr, title):
    fig, ax = plt.subplots()
    t = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(t, audio)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

# ------------------ Streamlit UI ------------------

st.title("🎤 Speech Preprocessing (Mic/File ➜ Filtered Output)")

# Session state to hold mic data
if "recorded_audio" not in st.session_state:
    st.session_state.recorded_audio = []

# Mic Audio Processor
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten()
        self.frames.append(audio)
        return frame

# Input Options
input_method = st.radio("Select input method:", ["Upload WAV file", "Record from Microphone (Browser)"])

sr = 48000

if input_method == "Upload WAV file":
    uploaded = st.file_uploader("Upload .wav file", type=["wav"])
    if uploaded is not None:
        audio, sr = sf.read(uploaded)
        if audio.ndim > 1:
            audio = audio[:, 0]

        st.subheader("🎧 Input Audio")
        st.audio(uploaded)
        plot_waveform(audio, sr, "Input Waveform")

        # Process
        filtered = apply_fir_bandpass(audio, sr)
        denoised = reduce_noise(filtered, sr)
        cleaned = normalize_audio(denoised)

        st.subheader("🧼 Cleaned Output")
        buf_out = BytesIO()
        sf.write(buf_out, cleaned, sr, format='wav')
        st.audio(buf_out)
        plot_waveform(cleaned, sr, "Output Waveform")

elif input_method == "Record from Microphone (Browser)":
    st.subheader("🎙️ Step 1: Record Mic Input")
    webrtc_ctx = webrtc_streamer(
        key="mic",
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    if st.button("🔁 Process Last 10 Seconds"):
        if webrtc_ctx.audio_processor:
            # Combine recent frames
            audio = np.concatenate(webrtc_ctx.audio_processor.frames)[-sr * 10:]
            st.audio(sf.write(BytesIO(), audio, sr, format='wav').getvalue())
            plot_waveform(audio, sr, "Input Mic Waveform")

            # Filter
            filtered = apply_fir_bandpass(audio, sr)
            denoised = reduce_noise(filtered, sr)
            cleaned = normalize_audio(denoised)

            st.subheader("🧼 Cleaned Output")
            buffer = BytesIO()
            sf.write(buffer, cleaned, sr, format='wav')
            st.audio(buffer)
            plot_waveform(cleaned, sr, "Output Mic Output")
        else:
            st.warning("⚠️ Please record audio first before clicking process.")


