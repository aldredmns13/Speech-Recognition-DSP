import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import numpy as np
import soundfile as sf
from io import BytesIO
import matplotlib.pyplot as plt
import noisereduce as nr

# ---------- DSP Helpers ----------

def reduce_noise(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr)

def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    return audio if max_val == 0 else audio / max_val * 0.9

def amplify_audio(audio, gain=1.5):
    audio_amp = audio * gain
    return np.clip(audio_amp, -1.0, 1.0)

def plot_waveform(audio, sr, title):
    fig, ax = plt.subplots()
    t = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(t, audio)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

# ---------- UI ----------

st.title("🎤 Voice Recorder & Enhancer (Mic Input)")

if "start_recording" not in st.session_state:
    st.session_state.start_recording = False

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten()
        # Convert to float32 from int16 or int
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0
        self.frames.append(audio)
        return frame

# Mic recording UI
st.subheader("🎙️ Step 1: Record Your Voice")

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

    st.info("Recording from mic... speak clearly for at least 10 seconds.")

    if st.button("✅ Process Last 10 Seconds"):
        if webrtc_ctx and webrtc_ctx.audio_processor:
            sr = 48000
            raw_audio = np.concatenate(webrtc_ctx.audio_processor.frames)

            if len(raw_audio) < sr * 2:
                st.warning("You need at least a few seconds of audio.")
            else:
                raw_audio = raw_audio[-sr * 10:]

                # ----- Original Audio -----
                st.subheader("🔊 Original Mic Audio")
                buf_orig = BytesIO()
                sf.write(buf_orig, raw_audio, sr, format='wav')
                st.audio(buf_orig)
                plot_waveform(raw_audio, sr, "Original Mic Waveform")

                # ----- Enhanced Audio -----
                st.subheader("🧼 Enhanced Mic Audio (Clearer)")
                cleaned = raw_audio
                cleaned = normalize_audio(cleaned)
                cleaned = reduce_noise(cleaned, sr)
                cleaned = amplify_audio(cleaned)
                cleaned = normalize_audio(cleaned)

                buf_clean = BytesIO()
                sf.write(buf_clean, cleaned.astype(np.float32), sr, format='wav')
                st.audio(buf_clean)
                plot_waveform(cleaned, sr, "Enhanced Mic Waveform")

        else:
            st.warning("Recording not started or no audio collected.")

