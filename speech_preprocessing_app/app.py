import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import noisereduce as nr
from scipy.signal import butter, lfilter
from io import BytesIO
import pandas as pd

# --- DSP Functions ---

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
    return np.clip(audio * gain, -1.0, 1.0)

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

# --- Streamlit UI ---

st.title("🎧 Speech Preprocessing App (Upload Only)")

uploaded_file = st.file_uploader("Upload a WAV file (mono, 16-bit PCM preferred)", type=["wav"])
TARGET_SR = 44100

if uploaded_file:
    audio, sr = librosa.load(uploaded_file, sr=None, mono=True)

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    st.subheader("🎧 Original Audio")
    st.audio(uploaded_file)
    plot_waveform(audio, sr, "Original Audio")

    # --- Processing Pipeline ---
    audio = normalize_audio(audio)
    audio = reduce_noise(audio, sr)
    audio = bandpass_filter(audio, sr)
    audio = amplify_audio(audio)
    audio = normalize_audio(audio)

    # --- Playback + Save Cleaned ---
    st.subheader("🧼 Cleaned Audio")
    buf_out = BytesIO()
    sf.write(buf_out, audio, sr, format='wav')
    st.audio(buf_out)
    plot_waveform(audio, sr, "Cleaned Audio")

    # --- MFCC ---
    mfcc = extract_mfcc(audio, sr)
    st.subheader("📊 MFCC Features")
    plot_mfcc(mfcc, sr)
    mfcc_df = pd.DataFrame(mfcc.T)
    st.download_button("Download MFCC CSV", mfcc_df.to_csv(index=False).encode(), "mfcc_features.csv")

    st.download_button("Download Cleaned Audio", buf_out.getvalue(), "cleaned_audio.wav", mime="audio/wav")

