import streamlit as st
from streamlit_webrtc import webrtc_streamer
import numpy as np
import noisereduce as nr
import av
import matplotlib.pyplot as plt

st.title("Live Mic Noise Reduction with streamlit-webrtc")

# This buffer stores audio chunks for plotting
audio_buffer = []

def audio_frame_callback(frame):
    audio = frame.to_ndarray(format="flt32")
    reduced_noise = nr.reduce_noise(y=audio.flatten(), sr=48000)

    # Save some audio to buffer for plotting
    audio_buffer.append(reduced_noise)

    new_frame = av.AudioFrame.from_ndarray(reduced_noise.astype(np.float32), format="flt32", layout="mono")
    new_frame.sample_rate = 48000
    return new_frame

webrtc_ctx = webrtc_streamer(key="mic", audio_frame_callback=audio_frame_callback, media_stream_constraints={"audio": True, "video": False})

if st.button("Show waveform of last audio chunk") and audio_buffer:
    audio_data = audio_buffer[-1]
    fig, ax = plt.subplots()
    time = np.linspace(0, len(audio_data) / 48000, num=len(audio_data))
    ax.plot(time, audio_data)
    ax.set_title("Noise Reduced Audio Waveform")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

