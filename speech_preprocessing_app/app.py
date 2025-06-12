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

# --- DSP Helpers ---

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
    fig, ax = plt.subplots()
    t = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(t, audio)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

# --- Mic Recorder with Real-Time Denoising ---

class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().mean(axis=0).astype(np.float32) / 32768.0  # mono
        self.frames.append(audio)

        # Real-time playback with light denoise
        denoised = reduce_noise(audio, sr=16000)
        denoised = normalize_audio(amplify_audio(denoised))

        int16_audio = (denoised * 32767).astype(np.int16)
        out_frame = av.AudioFrame.from_ndarray(int16_audio, layout="mono")
        out_frame.sample_rate = 16000
        return out_frame

# --- Streamlit UI ---

st.title("🎤 Speech Preprocessing (Mic + File Upload + Real-time Denoise)")
sr = 16000
input_method = st.radio("Choose input method:", ["🎙 Record via Microphone", "📁 Upload WAV File"])

def process_and_display(audio, sr):
    st.subheader("🔊 Original Audio")
    buf_orig = BytesIO()
    sf.write(buf_orig, audio, sr, format='wav')
    st.audio(buf_orig)
    plot_waveform(audio, sr, "Original Audio")

    audio = normalize_audio(audio)
    audio = reduce_noise(audio, sr)
    audio = bandpass_filter(audio, sr)
    audio = amplify_audio(audio)
    audio = normalize_audio(audio)

    st.subheader("🧼 Cleaned Audio")
    buf_clean = BytesIO()
    sf.write(buf_clean, audio, sr, format='wav')
    st.audio(buf_clean)
    plot_waveform(audio, sr, "Cleaned Audio")
    st.download_button("⬇️ Download Cleaned Audio", buf_clean.getvalue(), "cleaned_audio.wav", mime="audio/wav")

if input_method == "📁 Upload WAV File":
    uploaded = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded:
        y, file_sr = librosa.load(uploaded, sr=sr, mono=True)
        process_and_display(y, sr)

elif input_method == "🎙 Record via Microphone":
    st.info("Start speaking, then press Process after ~5–10 seconds.")

    webrtc_ctx = webrtc_streamer(
        key="mic",
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=False,
    )

    if st.button("✅ Process Mic Recording"):
        if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.audio_processor and webrtc_ctx.audio_processor.frames:
            raw_audio = np.concatenate(webrtc_ctx.audio_processor.frames)
            if len(raw_audio) < sr * 2:
                st.warning("Please record at least 2 seconds.")
            else:
                audio = raw_audio[-sr * 10:]  # last 10 seconds
                process_and_display(audio, sr)
        else:
            st.warning("No audio data available. Try speaking and then press the button.")
