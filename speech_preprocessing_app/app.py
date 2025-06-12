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

# DSP Helpers

def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    return audio if max_val == 0 else audio / max_val * 0.9

def reduce_noise(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr)

def bandpass_filter(audio, sr, lowcut=300.0, highcut=3400.0):
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

# Audio Processor

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32768.0  # Convert int16 to float32
        self.frames.append(audio)
        return frame

# Streamlit UI

st.title("🎙️ Mic Recorder: Clean Speech Processing")

sr = 48000
st.markdown("Click start and speak clearly. You can process the last 10 seconds.")

webrtc_ctx = webrtc_streamer(
    key="clean-mic",
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

if st.button("✅ Process Last 10 Seconds"):
    if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.audio_processor:
        raw_audio = np.concatenate(webrtc_ctx.audio_processor.frames)

        if len(raw_audio) < sr * 5:
            st.warning("Please speak for at least 5 seconds.")
        else:
            raw_audio = raw_audio[-sr * 10:]

            st.subheader("🎧 Original Mic Audio")
            buf_orig = BytesIO()
            sf.write(buf_orig, raw_audio, sr, format='wav')
            st.audio(buf_orig)
            plot_waveform(raw_audio, sr, "Original Mic Waveform")

            # Process pipeline
            audio = normalize_audio(raw_audio)
            audio = reduce_noise(audio, sr)
            audio = bandpass_filter(audio, sr)
            audio = amplify_audio(audio)
            audio = normalize_audio(audio)

            st.subheader("🧼 Cleaned & Enhanced Audio")
            buf_clean = BytesIO()
            sf.write(buf_clean, audio, sr, format='wav')
            st.audio(buf_clean)
            plot_waveform(audio, sr, "Cleaned Mic Waveform")

            st.download_button("Download Cleaned Audio", buf_clean.getvalue(), "cleaned_audio.wav", mime="audio/wav")
    else:
        st.warning("Recording hasn't started or no audio available.")
