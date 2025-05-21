import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import firwin, lfilter
from io import BytesIO
import noisereduce as nr
import sounddevice as sd

st.title("Speech Preprocessing App (Mic/File ➜ Filtered Output)")

# ------------------ Audio Processing Functions ------------------

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

# ------------------ UI: Input Options ------------------

input_method = st.radio("Select input method:", ["Upload WAV file", "Record from Microphone"])

sr = 44100  # Sample rate

if input_method == "Upload WAV file":
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    
    if uploaded_file is not None:
        audio, sr = sf.read(uploaded_file)
        if audio.ndim > 1:
            audio = audio[:, 0]  # Convert to mono

        st.subheader("🎧 Original Audio (Input)")
        st.audio(uploaded_file)
        plot_waveform(audio, sr, "Input Waveform")

        # Process
        filtered = apply_fir_bandpass(audio, sr)
        denoised = reduce_noise(filtered, sr)
        cleaned = normalize_audio(denoised)

        st.subheader("🧼 Cleaned Audio (Output)")
        cleaned_buffer = BytesIO()
        sf.write(cleaned_buffer, cleaned, sr, format='wav')
        st.audio(cleaned_buffer)
        plot_waveform(cleaned, sr, "Output Waveform")

elif input_method == "Record from Microphone":
    duration = 10  # seconds
    if st.button("🎙️ Record 10 Seconds"):
        st.info("Recording... Speak clearly into your mic.")
        recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
        sd.wait()
        audio = recording.flatten()

        st.success("Recording complete!")

        st.subheader("🎧 Original Mic Recording (Input)")
        buffer = BytesIO()
        sf.write(buffer, audio, sr, format='wav')
        st.audio(buffer)
        plot_waveform(audio, sr, "Input Mic Waveform")

        # Process
        filtered = apply_fir_bandpass(audio, sr)
        denoised = reduce_noise(filtered, sr)
        cleaned = normalize_audio(denoised)

        st.subheader("🧼 Cleaned Mic Audio (Output)")
        out_buffer = BytesIO()
        sf.write(out_buffer, cleaned, sr, format='wav')
        st.audio(out_buffer)
        plot_waveform(cleaned, sr, "Output Mic Waveform")
