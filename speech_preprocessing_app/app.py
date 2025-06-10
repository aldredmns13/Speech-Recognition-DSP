import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import numpy as np
import soundfile as sf
from io import BytesIO
import matplotlib.pyplot as plt

# ---------- Basic Helpers ----------

def plot_waveform(audio, sr, title):
    fig, ax = plt.subplots()
    t = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(t, audio)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

# ---------- UI ----------

st.title("🎙️ Simple Voice Recorder (Mic Only)")

# Maintain state
if "start_recording" not in st.session_state:
    st.session_state.start_recording = False

# Audio processor (no processing, just collect audio)
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten()
        # Convert from int16 to float32 in [-1.0, 1.0] range
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0
        self.frames.append(audio)
        return frame

# Start mic
st.subheader("🎤 Record from Microphone")

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

    st.info("Recording from your mic... speak clearly for about 10 seconds.")

    if st.button("✅ Save Last 10 Seconds"):
        if webrtc_ctx and webrtc_ctx.audio_processor:
            sr = 48000  # Make sure this matches WebRTC audio
            raw_audio = np.concatenate(webrtc_ctx.audio_processor.frames)

            if len(raw_audio) < sr * 2:
                st.warning("You need at least a few seconds of recording.")
            else:
                raw_audio = raw_audio[-sr * 10:]  # Last 10 seconds

                st.subheader("🔊 Playback")
                buffer = BytesIO()
                sf.write(buffer, raw_audio.astype(np.float32), sr, format='wav')
                st.audio(buffer)

                plot_waveform(raw_audio, sr, "Mic Recording Waveform")

        else:
            st.warning("Recording not started or no audio captured.")

