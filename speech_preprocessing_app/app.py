import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import numpy as np
import soundfile as sf
from io import BytesIO
import matplotlib.pyplot as plt
import noisereduce as nr
from scipy.signal import resample

# ---------- Helpers ----------

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

st.title("🎤 Mic Voice Recorder & Enhancer (No Reverb, Normal Voice)")

if "start_recording" not in st.session_state:
    st.session_state.start_recording = False

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        self.buffer.append(audio)
        return frame

st.subheader("🎙️ Step 1: Record from Microphone")

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

    st.info("Recording... speak clearly for 5–10 seconds.")

    if st.button("✅ Process Last 10 Seconds"):
        if webrtc_ctx and webrtc_ctx.audio_processor:
            sr = 48000
            frames = webrtc_ctx.audio_processor.buffer

            if not frames or len(frames) < 10:
                st.warning("Not enough audio captured yet.")
            else:
                raw_audio = np.concatenate(frames)

                # Get last 10 seconds (or less)
                max_samples = sr * 10
                if len(raw_audio) > max_samples:
                    raw_audio = raw_audio[-max_samples:]

                # Downsample to 16000 Hz for clarity
                target_sr = 16000
                raw_downsampled = resample(raw_audio, int(len(raw_audio) * target_sr / sr))

                st.subheader("🔊 Original Mic Audio")
                buf_in = BytesIO()
                sf.write(buf_in, raw_downsampled, target_sr, format='wav')
                st.audio(buf_in)
                plot_waveform(raw_downsampled, target_sr, "Original Mic Waveform")

                # Enhance: denoise + amplify + normalize
                enhanced = reduce_noise(raw_downsampled, target_sr)
                enhanced = amplify_audio(enhanced)
                enhanced = normalize_audio(enhanced)

                st.subheader("🧼 Enhanced Mic Audio (Clear & Normal)")
                buf_out = BytesIO()
                sf.write(buf_out, enhanced, target_sr, format='wav')
                st.audio(buf_out)
                plot_waveform(enhanced, target_sr, "Enhanced Mic Waveform")

        else:
            st.warning("Mic not recording yet.")
