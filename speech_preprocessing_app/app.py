import streamlit as st
import numpy as np
import soundfile as sf
import sounddevice as sd
import noisereduce as nr  # pip install noisereduce
import matplotlib.pyplot as plt
from io import BytesIO

st.title("Speech Preprocessing Web App with Noise Reduction and Mic Input")

# Option to use mic or upload
input_option = st.radio("Select input source:", ["Upload audio file", "Record from microphone"])

def plot_waveform(audio, sr, title="Waveform"):
    fig, ax = plt.subplots()
    time = np.linspace(0, len(audio) / sr, num=len(audio))
    ax.plot(time, audio)
    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

if input_option == "Upload audio file":
    uploaded_file = st.file_uploader("Upload audio (wav format)", type=["wav"])
    if uploaded_file is not None:
        audio, sr = sf.read(uploaded_file)
        st.audio(uploaded_file)
        
        plot_waveform(audio, sr, "Original Audio")

        # Apply noise reduction using noisereduce
        reduced_noise = nr.reduce_noise(y=audio, sr=sr)

        st.audio(BytesIO(sf.write(BytesIO(), reduced_noise, sr, format='wav').getbuffer()))

        plot_waveform(reduced_noise, sr, "Noise Reduced Audio")

elif input_option == "Record from microphone":
    duration = st.slider("Recording duration (seconds)", 1, 10, 3)
    if st.button("Record"):
        st.info("Recording...")
        audio = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
        sd.wait()
        audio = audio.flatten()

        plot_waveform(audio, 44100, "Original Mic Recording")

        reduced_noise = nr.reduce_noise(y=audio, sr=44100)

        plot_waveform(reduced_noise, 44100, "Noise Reduced Mic Audio")

        # Save to buffer and play
        buffer = BytesIO()
        sf.write(buffer, reduced_noise, 44100, format='wav')
        st.audio(buffer)

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import numpy as np
import noisereduce as nr
import av

def audio_frame_callback(frame):
    audio = frame.to_ndarray(format="flt32")
    reduced_noise = nr.reduce_noise(y=audio.flatten(), sr=48000)
    # Convert back to frame (do minimal processing here to keep real-time)
    new_frame = av.AudioFrame.from_ndarray(reduced_noise.astype(np.float32), format="flt32", layout="mono")
    new_frame.sample_rate = 48000
    return new_frame

st.title("Mic Noise Reduction with streamlit-webrtc")

webrtc_streamer(key="mic", audio_frame_callback=audio_frame_callback, media_stream_constraints={"audio": True, "video": False})
