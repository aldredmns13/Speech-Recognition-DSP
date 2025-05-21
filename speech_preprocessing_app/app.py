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
    # Mild noise reduction to preserve natural voice
    return nr.reduce_noise(y=audio, sr=sr)

def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return audio
    return audio / max_val * 0.9

def amplify_audio(audio, gain=2.0):
    # Apply gain and clip to avoid distortion
    audio_amp = audio * gain
    audio_amp = np.clip(audio_amp, -1.0, 1.0)
    return audio_amp

def plot_waveform(audio, sr, title):
    fig, ax = plt.subplots()
    t = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(t, audio)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

# ------------------ UI ------------------

st.title("🎤 Speech Preprocessing Module (Mic or File Input → Enhanced Audio)")

# Session state setup
if "start_recording" not in st.session_state:
    st.session_state.start_recording = False

# Mic Audio Processor
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten()
        self.frames.append(audio)
        return frame

# Choose input method
input_method = st.radio("Select Input Source", ["Upload .wav File", "Record via Microphone (Browser)"])

sr = 48000

if input_method == "Upload .wav File":
    uploaded = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded:
        audio, sr = sf.read(uploaded)
        if audio.ndim > 1:
            audio = audio[:, 0]

        st.subheader("🎧 Original Audio")
        st.audio(uploaded)
        plot_waveform(audio, sr, "Original Audio Waveform")

        filtered = apply_fir_bandpass(audio, sr)
        denoised = reduce_noise(filtered, sr)
        cleaned = normalize_audio(denoised)

        st.subheader("🧼 Cleaned Audio")
        buf_out = BytesIO()
        sf.write(buf_out, cleaned, sr, format='wav')
        st.audio(buf_out)
        plot_waveform(cleaned, sr, "Cleaned Audio Waveform")

elif input_method == "Record via Microphone (Browser)":
    st.subheader("🎙️ Step 1: Start Recording")

    if not st.session_state.start_recording:
        if st.button("🎙️ Start Mic"):
            st.session_state.start_recording = True
            st.experimental_rerun()

    if st.session_state.start_recording:
        webrtc_ctx = webrtc_streamer(
            key="mic",
            audio_processor_factory=AudioProcessor,
            media_stream_constraints={"audio": True, "video": False},
            async_processing=True,
        )

        st.info("Recording from browser mic... speak clearly for at least 10 seconds.")

        if st.button("✅ Process Last 10 Seconds"):
            if webrtc_ctx and webrtc_ctx.audio_processor:
                raw_audio = np.concatenate(webrtc_ctx.audio_processor.frames)
                if len(raw_audio) < sr * 10:
                    st.warning("You need at least 10 seconds of speech.")
                else:
                    raw_audio = raw_audio[-sr * 10:]

                    st.subheader("🎧 Original Mic Audio")
                    buffer_in = BytesIO()
                    sf.write(buffer_in, raw_audio, sr, format='wav')
                    st.audio(buffer_in)
                    plot_waveform(raw_audio, sr, "Original Mic Waveform")

                    # === Enhanced natural voice processing ===
                    audio_norm = normalize_audio(raw_audio)          # Soft normalization
                    audio_denoised = reduce_noise(audio_norm, sr)    # Mild noise reduction
                    audio_amplified = amplify_audio(audio_denoised)  # Gentle amplification x2
                    cleaned = normalize_audio(audio_amplified)       # Final normalization
                    # ==============================================

                    st.subheader("🧼 Enhanced Mic Audio")
                    buffer_out = BytesIO()
                    sf.write(buffer_out, cleaned, sr, format='wav')
                    st.audio(buffer_out)
                    plot_waveform(cleaned, sr, "Enhanced Mic Waveform")
            else:
                st.warning("Recording has not started or no frames available.")
