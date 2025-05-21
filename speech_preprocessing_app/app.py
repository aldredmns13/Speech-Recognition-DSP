import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import noisereduce as nr

# Sampling parameters
SAMPLE_RATE = 16000  # 16 kHz sample rate
DURATION = 5  # seconds to record

def normalize_audio(audio):
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    return audio / peak

def highpass_filter(audio, sr, cutoff=100, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_audio = lfilter(b, a, audio)
    return filtered_audio

def plot_waveform(audio, sr, title):
    times = np.arange(len(audio)) / sr
    plt.figure(figsize=(10, 3))
    plt.plot(times, audio)
    plt.title(title)
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    plt.xlim(0, len(audio) / sr)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

st.title("Speech Preprocessing Module")

if st.button("Record Audio from Mic (5 seconds)"):
    st.info("Recording...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()  # Wait until recording is finished
    audio = audio.flatten()
    st.success("Recording complete!")

    # Show raw audio waveform
    plot_waveform(audio, SAMPLE_RATE, "Raw Audio")

    # Normalize
    audio_norm = normalize_audio(audio)

    # High-pass filter to remove low-frequency rumble
    audio_hp = highpass_filter(audio_norm, SAMPLE_RATE, cutoff=100)

    # Noise reduction (optional, comment if you want)
    audio_clean = nr.reduce_noise(y=audio_hp, sr=SAMPLE_RATE)

    # Normalize again after noise reduction
    audio_clean = normalize_audio(audio_clean)

    # Show processed audio waveform
    plot_waveform(audio_clean, SAMPLE_RATE, "Processed Audio")

    st.audio(audio_clean, format="audio/wav", sample_rate=SAMPLE_RATE)

    # Optionally save the processed audio to a file for MATLAB use
    # from scipy.io.wavfile import write
    # write("processed_audio.wav", SAMPLE_RATE, (audio_clean * 32767).astype(np.int16))
