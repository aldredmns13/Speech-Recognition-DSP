import streamlit as st
import numpy as np
import soundfile as sf
import noisereduce as nr
import os
import tempfile

try:
    import matlab.engine
    has_matlab = True
except ImportError:
    has_matlab = False

st.set_page_config(page_title="Speech Recognition Preprocessing", layout="centered")

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

st.title("🎙️ Speech Recognition Preprocessing Module")
st.markdown("Upload a `.wav` file, denoise it, and visualize the waveform using MATLAB (optional).")

uploaded_file = st.file_uploader("📁 Upload .wav Audio File", type=["wav"])

if uploaded_file:
    audio_data, fs = sf.read(uploaded_file)
    original_path = os.path.join(tempfile.gettempdir(), "original_file.wav")
    save_audio_file(original_path, audio_data, fs)

    denoised_audio = denoise_audio(audio_data, fs)
    filtered_path = os.path.join(tempfile.gettempdir(), "filtered_file.wav")
    save_audio_file(filtered_path, denoised_audio, fs)

    st.success("Audio successfully filtered!")

    st.audio(original_path, format="audio/wav", start_time=0)
    st.audio(filtered_path, format="audio/wav", start_time=0)

    if has_matlab and st.button("📊 Show MATLAB Sine Waves"):
        fig_path = run_matlab_plot(original_path, filtered_path)
        st.image(fig_path, caption="Original vs Filtered (MATLAB)")
    elif not has_matlab:
        st.warning("MATLAB not available. Plotting is disabled.")
