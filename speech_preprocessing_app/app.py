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

# --- Audio DSP Functions ---

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
    times = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(times, audio)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

def plot_mfcc(mfcc, sr):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    ax.set_title("MFCC Features")
    fig.colorbar(img, ax=ax)
    st.pyplot(fig)

# --- Streamlit App UI ---

st.title("🧠 Speech Preprocessing App")

input_method = st.radio("Select Input Method", ["Upload .wav File", "Record via Microphone (Browser)"])
sr = 48000

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten()

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        self.frames.append(audio)
        return frame

audio = None
source_label = ""

if input_method == "Upload .wav File":
    uploaded = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded:
        audio, sr = sf.read(uploaded)
        if audio.ndim > 1:
            audio = audio[:, 0]
        source_label = "Uploaded Audio"
        st.subheader("🎧 Original Uploaded Audio")
        st.audio(uploaded)
        plot_waveform(audio, sr, "Original Audio Waveform")

elif input_method == "Record via Microphone (Browser)":
    if "start_recording" not in st.session_state:
        st.session_state.start_recording = False

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

        st.info("Recording... Speak clearly for at least 10 seconds.")

        if st.button("✅ Process Mic Audio"):
            if webrtc_ctx and webrtc_ctx.audio_processor:
                audio = np.concatenate(webrtc_ctx.audio_processor.frames)
                if len(audio) < sr * 2:
                    st.warning("Please record at least 2 seconds.")
                else:
                    audio = audio[-sr * 10:]
                    source_label = "Recorded Mic Audio"
                    st.subheader("🎧 Original Mic Audio")
                    buffer_in = BytesIO()
                    sf.write(buffer_in, audio, sr, format='wav')
                    st.audio(buffer_in)
                    plot_waveform(audio, sr, "Original Mic Waveform")

# --- Process and Output ---

if audio is not None:
    # Step 1: Normalize
    norm = normalize_audio(audio)

    # Step 2: Noise Reduction
    denoised = reduce_noise(norm, sr)

    # Step 3: Bandpass Filtering
    filtered = bandpass_filter(denoised, sr)

    # Step 4: MFCC Extraction
    mfcc = extract_mfcc(filtered, sr)

    # Step 5: Output Results
    st.subheader("🧼 Cleaned Audio Output")
    cleaned_buf = BytesIO()
    sf.write(cleaned_buf, filtered, sr, format='wav')
    st.audio(cleaned_buf)
    plot_waveform(filtered, sr, "Cleaned Audio Waveform")

    st.subheader("📊 MFCC Features")
    plot_mfcc(mfcc, sr)

    # Optional: download buttons
    st.download_button("📥 Download Cleaned Audio", data=cleaned_buf.getvalue(), file_name="cleaned_audio.wav", mime="audio/wav")

    mfcc_df = pd.DataFrame(mfcc).T
    mfcc_csv = mfcc_df.to_csv(index=False).encode()
    st.download_button("📥 Download MFCC CSV", data=mfcc_csv, file_name="mfcc_features.csv", mime="text/csv")
