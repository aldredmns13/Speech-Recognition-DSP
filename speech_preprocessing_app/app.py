import streamlit as st
import numpy as np
import soundfile as sf
import sounddevice as sd
import noisereduce as nr
import os
import matlab.engine
from io import BytesIO
import tempfile
from PIL import Image

st.set_page_config(page_title="Speech Recognition Preprocessing Module", layout="centered")

def record_audio(duration=5, fs=44100):
    st.info(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return audio, fs

def save_audio_file(file_path, data, samplerate):
    sf.write(file_path, data, samplerate)

def denoise_audio(data, fs):
    reduced_noise = nr.reduce_noise(y=data.flatten(), sr=fs)
    return reduced_noise

def run_matlab_plot(original_path, filtered_path):
    eng = matlab.engine.start_matlab()
    fig_path = eng.plot_audio_waves(original_path, filtered_path, nargout=1)
    eng.quit()
    return fig_path

st.title("Speech Recognition Preprocessing Module")
st.subheader("Clean your noisy Audio with Us!")

option = st.radio("Select Input Method", ("Insert Audio from Mic", "Insert Audio from Files"))

if option == "Insert Audio from Mic":
    if st.button("🎤 Start Recording"):
        audio, fs = record_audio()
        original_path = os.path.join(tempfile.gettempdir(), "original_mic.wav")
        save_audio_file(original_path, audio, fs)

        denoised_audio = denoise_audio(audio, fs)
        filtered_path = os.path.join(tempfile.gettempdir(), "filtered_mic.wav")
        save_audio_file(filtered_path, denoised_audio, fs)

        st.success("Audio recorded and filtered!")

        if st.button("📈 Show MATLAB Sine Waves"):
            fig_path = run_matlab_plot(original_path, filtered_path)
            st.image(fig_path, caption="Original vs Filtered (Mic Input)")

elif option == "Insert Audio from Files":
    uploaded_file = st.file_uploader("Upload .wav File", type=["wav"])
    if uploaded_file:
        audio_data, fs = sf.read(uploaded_file)
        original_path = os.path.join(tempfile.gettempdir(), "original_file.wav")
        save_audio_file(original_path, audio_data, fs)

        denoised_audio = denoise_audio(audio_data, fs)
        filtered_path = os.path.join(tempfile.gettempdir(), "filtered_file.wav")
        save_audio_file(filtered_path, denoised_audio, fs)

        st.success("Audio uploaded and filtered!")

        if st.button("📈 Show MATLAB Sine Waves"):
            fig_path = run_matlab_plot(original_path, filtered_path)
            st.image(fig_path, caption="Original vs Filtered (File Input)")
