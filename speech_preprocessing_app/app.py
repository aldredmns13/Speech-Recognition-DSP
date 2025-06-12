import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display
from io import BytesIO
from scipy.signal import butter, lfilter
import noisereduce as nr
import pandas as pd

# --- DSP Helpers ---

def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    return audio if max_val == 0 else audio / max_val * 0.9

def reduce_noise(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr)

def bandpass_filter(audio, sr, lowcut=300.0, highcut=3400.0, order=6):
    nyq = 0.5 * sr
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return lfilter(b, a, audio)

def extract_mfcc(audio, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def plot_waveform(audio, sr, title="Waveform"):
    fig, ax = plt.subplots()
    t = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(t, audio)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

def plot_mfcc(mfcc, sr):
    fig, ax = plt.subplots()
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    ax.set_title("MFCC Features")
    st.pyplot(fig)

# --- Streamlit UI ---

st.title("🔊 Speech Preprocessing with MFCC + Enhancements")

st.markdown("""
- ✅ Uploading WAV is **recommended** for clean quality.
- 🎙️ Mic recording is **experimental** and may vary in quality depending on your browser/mic.
""")

input_method = st.radio("Choose Input Method", ["Upload .wav File (Recommended)", "Record via Microphone (Experimental)"])
TARGET_SR = 44100

# --- AudioProcessor for Mic Input ---

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten()

        # Normalize to float32 in [-1, 1]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        self.frames.append(audio)
        return frame

# --- Audio Source Handling ---

audio = None
actual_sr = None
source_label = ""

if input_method.startswith("Upload"):
    uploaded = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded:
        y, actual_sr = librosa.load(uploaded, sr=None, mono=True)
        st.write(f"🎯 Original file sample rate: {actual_sr} Hz")
        if actual_sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=actual_sr, target_sr=TARGET_SR)
        audio = y
        source_label = "Uploaded Audio"
        st.subheader("🎧 Original Uploaded Audio")
        st.audio(uploaded)
        plot_waveform(audio, TARGET_SR, "Original Audio Waveform")

elif input_method.startswith("Record"):
    if "start_recording" not in st.session_state:
        st.session_state.start_recording = False

    if not st.session_state.start_recording:
        if st.button("🎙️ Start Recording"):
            st.session_state.start_recording = True
            st.experimental_rerun()

    if st.session_state.start_recording:
        webrtc_ctx = webrtc_streamer(
            key="mic",
            audio_processor_factory=AudioProcessor,
            media_stream_constraints={"audio": {"sampleRate": TARGET_SR}, "video": False},
            async_processing=True,
        )

        st.info("Recording... Speak for ~5–10 seconds.")

        if st.button("✅ Process Mic Audio"):
            if webrtc_ctx and webrtc_ctx.audio_processor:
                raw_audio = np.concatenate(webrtc_ctx.audio_processor.frames)

                if len(raw_audio) < TARGET_SR * 2:
                    st.warning("Please record for at least 2 seconds.")
                else:
                    raw_audio = raw_audio[-TARGET_SR * 10:]  # Use last 10 seconds
                    audio = raw_audio
                    source_label = "Recorded Mic Audio"
                    actual_sr = TARGET_SR
                    st.write(f"🎯 Using sample rate: {TARGET_SR} Hz")

                    # Optional: Save + reload to validate
                    buffer_test = BytesIO()
                    sf.write(buffer_test, audio, TARGET_SR, format='wav')
                    y, real_sr = librosa.load(BytesIO(buffer_test.getvalue()), sr=None)
                    st.write(f"📏 Detected actual sample rate after save: {real_sr} Hz")

                    st.subheader("🎧 Original Mic Audio")
                    st.audio(buffer_test)
                    plot_waveform(audio, TARGET_SR, "Original Mic Waveform")

# --- Process Pipeline ---

if audio is not None:
    st.subheader("🧠 Processing Pipeline")
    norm = normalize_audio(audio)
    denoised = reduce_noise(norm, TARGET_SR)
    filtered = bandpass_filter(denoised, TARGET_SR)
    mfcc = extract_mfcc(filtered, TARGET_SR)

    # -- Output cleaned audio --
    st.subheader("🧼 Cleaned Audio Output")
    buf_out = BytesIO()
    sf.write(buf_out, filtered, TARGET_SR, format='wav')
    st.audio(buf_out)
    plot_waveform(filtered, TARGET_SR, "Cleaned Audio Waveform")

    # -- MFCC Visualization & Export --
    st.subheader("📊 MFCC Features")
    plot_mfcc(mfcc, TARGET_SR)

    st.download_button("📥 Download Cleaned Audio", data=buf_out.getvalue(), file_name="cleaned_audio.wav", mime="audio/wav")

    mfcc_df = pd.DataFrame(mfcc).T
    st.download_button("📥 Download MFCC Features CSV", data=mfcc_df.to_csv(index=False).encode(), file_name="mfcc_features.csv", mime="text/csv")
