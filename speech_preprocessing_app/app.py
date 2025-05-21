import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import firwin, lfilter
import sounddevice as sd
from io import BytesIO
import noisereduce as nr
import time

# ---------- Functions ----------
def apply_fir_bandpass(audio, sr, lowcut=300.0, highcut=3400.0, numtaps=101):
    fir_coeff = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=sr)
    return lfilter(fir_coeff, 1.0, audio)

def reduce_noise(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr)

def normalize_audio(audio):
    return audio / np.max(np.abs(audio)) * 0.9

def plot_waveform(audio, sr, title):
    fig, ax = plt.subplots()
    t = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(t, audio)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

# ---------- Streamlit App ----------
st.title("🎤 Real-Time Mic Recording with Audio Filtering")

sr = 44100
duration = 10  # seconds

if st.button("🎙️ Start Recording (10 seconds)"):
    st.info("Recording... Please speak clearly.")
    
    audio_buffer = np.zeros((int(duration * sr),))
    chunk_size = 1024
    total_chunks = int(duration * sr / chunk_size)
    stream_plot = st.empty()

    def callback(indata, frames, time_info, status):
        callback.audio_data.extend(indata[:, 0].tolist())  # Flatten mono
    callback.audio_data = []

    with sd.InputStream(callback=callback, channels=1, samplerate=sr, blocksize=chunk_size):
        for _ in range(total_chunks):
            if len(callback.audio_data) > chunk_size:
                # Display live waveform every chunk
                data_chunk = np.array(callback.audio_data[-sr:])  # last 1 sec
                fig, ax = plt.subplots()
                t = np.linspace(0, len(data_chunk) / sr, len(data_chunk))
                ax.plot(t, data_chunk)
                ax.set_title("🔴 Live Input Waveform")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                stream_plot.pyplot(fig)
            time.sleep(chunk_size / sr)

    st.success("✅ Recording complete!")

    audio = np.array(callback.audio_data)
    if len(audio) < duration * sr:
        # Zero-pad if too short
        audio = np.pad(audio, (0, int(duration * sr) - len(audio)))

    # Display input
    st.subheader("🎧 Raw Input Audio")
    buffer_in = BytesIO()
    sf.write(buffer_in, audio, sr, format='wav')
    st.audio(buffer_in)
    plot_waveform(audio, sr, "Input Audio Waveform")

    # Processing
    filtered = apply_fir_bandpass(audio, sr)
    denoised = reduce_noise(filtered, sr)
    cleaned = normalize_audio(denoised)

    # Display output
    st.subheader("🧼 Cleaned Audio Output")
    buffer_out = BytesIO()
    sf.write(buffer_out, cleaned, sr, format='wav')
    st.audio(buffer_out)
    plot_waveform(cleaned, sr, "Output Audio Waveform")

