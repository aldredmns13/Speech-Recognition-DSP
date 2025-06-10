import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import numpy as np
import soundfile as sf
from scipy.signal import resample
from io import BytesIO

st.title("🎙️ Normal Voice Recorder")

# Audio settings
TARGET_SR = 16000  # Force sample rate to 16 kHz

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        self.frames.append(audio)
        return frame

# UI
if "recording" not in st.session_state:
    st.session_state.recording = False

if not st.session_state.recording:
    if st.button("🎙️ Start Recording"):
        st.session_state.recording = True
        st.experimental_rerun()

if st.session_state.recording:
    ctx = webrtc_streamer(
        key="mic",
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    st.info("Recording... speak normally into the mic.")

    if st.button("✅ Save Last 10 Seconds"):
        if ctx and ctx.audio_processor:
            raw = np.concatenate(ctx.audio_processor.frames)
            sr = 48000  # WebRTC streams at 48k

            if len(raw) >= sr * 2:
                # Take last 10s
                raw = raw[-sr * 10:] if len(raw) > sr * 10 else raw

                # Downsample to 16kHz (for normal playback)
                audio_16k = resample(raw, int(len(raw) * TARGET_SR / sr))

                # Save to buffer
                buffer = BytesIO()
                sf.write(buffer, audio_16k, TARGET_SR, format="wav")

                st.subheader("▶️ Playback (Normal)")
                st.audio(buffer)
            else:
                st.warning("Speak a bit longer before saving.")
