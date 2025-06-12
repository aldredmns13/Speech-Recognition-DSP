import streamlit as st
from streamlit_audio_recorder import audio_recorder
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import noisereduce as nr
from io import BytesIO
from scipy.signal import butter, lfilter
import pandas as pd

# --- Helpers ---

def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    return audio if max_val == 0 else audio / max_val * 0.9

def reduce_noise(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr)

def bandpass_filter(audio, sr, lowcut=300.0, highcut=3400.0, order=6):
    nyq = 0.5 * sr
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return lfilter(b, a, audio)

def amplify_audio(audio, gain=2.0):
    amplified = audio * gain
    return np.clip(amplified, -1.0, 1.0)

def extract_mfcc(audio, sr, n_mfcc=13):
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

def plot_waveform(audio, sr, title):
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

# --- UI ---

st.title("🎙️ Voice Preprocessing & Feature Extraction")

input_method = st.radio("Select Input Source", ["Upload .wav File", "Record via Microphone"])

TARGET_SR = 44100
audio = None

# --- Input Option 1: Upload ---
if input_method == "Upload .wav File":
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file:
        audio, sr = librosa.load(uploaded_file, sr=None, mono=True)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        st.audio(uploaded_file)
        plot_waveform(audio, sr, "Uploaded Audio")

# --- Input Option 2: Mic ---
elif input_method == "Record via Microphone":
    audio_bytes = audio_recorder()
    if audio_bytes:
        # Convert to NumPy array
        with sf.SoundFile(BytesIO(audio_bytes)) as f:
            audio = f.read(dtype="float32")
            sr = f.samplerate
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        st.audio(audio_bytes)
        plot_waveform(audio, sr, "Recorded Mic Audio")

# --- Processing Pipeline ---
if audio is not None:
    st.subheader("🔧 Processing Pipeline")

    audio = normalize_audio(audio)
    audio = reduce_noise(audio, sr)
    audio = bandpass_filter(audio, sr)
    audio = amplify_audio(audio)
    audio = normalize_audio(audio)

    # --- Output cleaned audio ---
    st.subheader("🧼 Cleaned Audio")
    buffer_out = BytesIO()
    sf.write(buffer_out, audio, sr, format='wav')
    st.audio(buffer_out)
    plot_waveform(audio, sr, "Cleaned Waveform")

    # --- MFCC ---
    mfcc = extract_mfcc(audio, sr)
    st.subheader("📊 MFCC Features")
    plot_mfcc(mfcc, sr)
    mfcc_df = pd.DataFrame(mfcc.T)
    st.download_button("Download MFCC CSV", mfcc_df.to_csv(index=False).encode(), "mfcc_features.csv")

    st.download_button("Download Cleaned Audio", buffer_out.getvalue(), "cleaned_audio.wav", mime="audio/wav")
