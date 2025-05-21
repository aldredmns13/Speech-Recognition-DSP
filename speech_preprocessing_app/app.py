import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from io import BytesIO
from streamlit_webrtc import webrtc_streamer
import av
import noisereduce as nr
from scipy.signal import firwin, lfilter

st.title("Speech Preprocessing App with FIR Filtering")

# ----------------- Signal Processing Functions ------------------

def apply_fir_bandpass(audio, sr, lowcut=300.0, highcut=3400.0, numtaps=101):
    fir_coeff = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=sr)
    return lfilter(fir_coeff, 1.0, audio)

def reduce_noise(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr)

def normalize_audio(audio):
    return audio / np.max(np.abs(audio)) * 0.9

def plot_waveform(audio, sr, title):
    fig, ax = plt.subplots()
    time = np.linspace(0, len(audio) / sr, num=len(audio))
    ax.plot(time, audio)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

# ----------------- App UI ------------------

# To store last mic input
if "mic_audio" not in st.session_state:
    st.session_state.mic_audio = None

input_mode = st.radio("Select Input Mode", ["Upload WAV File", "Use Microphone"])

# ----------------- File Upload Mode ------------------
if input_mode == "Upload WAV File":
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file is not None:
        audio, sr = sf.read(uploaded_file)
        if audio.ndim > 1:
            audio = audio[:, 0]  # Convert stereo to mono

        st.subheader("Original Audio")
        st.audio(uploaded_file)
        plot_waveform(audio, sr, "Original Audio")

        st.subheader("Filtered + Denoised + Normalized Audio")

        filtered = apply_fir_bandpass(audio, sr)
        denoised = reduce_noise(filtered, sr)
        cleaned = normalize_audio(denoised)

        plot_waveform(cleaned, sr, "Cleaned Audio")
        cleaned_buffer = BytesIO()
        sf.write(cleaned_buffer, cleaned, sr, format='wav')
        st.audio(cleaned_buffer)

# ----------------- Microphone Mode ------------------
def audio_frame_callback(frame):
    audio = frame.to_ndarray(format="flt32").flatten()

    # FIR filtering + noise reduction + normalization
    filtered = apply_fir_bandpass(audio, sr=48000)
    denoised = reduce_noise(filtered, sr=48000)
    cleaned = normalize_audio(denoised)

    # Save cleaned audio for playback
    st.session_state.mic_audio = cleaned

    # Append to live waveform buffer
    buffer = st.session_state.live_audio_buffer
    buffer = np.roll(buffer, -len(cleaned))
    buffer[-len(cleaned):] = cleaned
    st.session_state.live_audio_buffer = buffer

    # Return cleaned audio frame
    new_frame = av.AudioFrame.from_ndarray(cleaned.astype(np.float32), format="flt32", layout="mono")
    new_frame.sample_rate = 48000
    return new_frame
import time

# Initialize the rolling buffer once
if "live_audio_buffer" not in st.session_state:
    st.session_state.live_audio_buffer = np.zeros(48000)

# Display mic stream
webrtc_streamer(
    key="mic",
    audio_frame_callback=audio_frame_callback,
    media_stream_constraints={"audio": True, "video": False}
)

# Real-time waveform plotting outside the callback
st.subheader("Real-Time Mic Audio Waveform")
plot_area = st.empty()

# Continuously update the plot for a few seconds
plot_duration = 20  # seconds
start_time = time.time()
while time.time() - start_time < plot_duration:
    buffer = st.session_state.live_audio_buffer
    fig, ax = plt.subplots()
    t = np.linspace(0, len(buffer) / 48000, len(buffer))
    ax.plot(t, buffer)
    ax.set_title("Real-Time Mic Audio")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plot_area.pyplot(fig)
    time.sleep(0.2)
