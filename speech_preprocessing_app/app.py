import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from io import BytesIO
from streamlit_webrtc import webrtc_streamer
import av
import noisereduce as nr

st.title("Speech Preprocessing App (Mic + File Upload)")

# State to store recorded audio
if "mic_audio" not in st.session_state:
    st.session_state.mic_audio = None

# Helper to plot waveform
def plot_waveform(audio, sr, title):
    fig, ax = plt.subplots()
    time = np.linspace(0, len(audio) / sr, num=len(audio))
    ax.plot(time, audio)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

# Noise reduction function
def reduce_noise(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr)

# --- Input Options ---
input_mode = st.radio("Select Input Mode", ["Upload WAV File", "Use Microphone"])

if input_mode == "Upload WAV File":
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file is not None:
        audio, sr = sf.read(uploaded_file)
        if audio.ndim > 1:
            audio = audio[:, 0]  # Convert to mono if stereo

        st.subheader("Original Audio")
        st.audio(uploaded_file)
        plot_waveform(audio, sr, "Original Audio")

        st.subheader("Noise Reduced Audio")
        denoised = reduce_noise(audio, sr)
        plot_waveform(denoised, sr, "Cleaned Audio")

        # Play cleaned audio
        cleaned_buffer = BytesIO()
        sf.write(cleaned_buffer, denoised, sr, format='wav')
        st.audio(cleaned_buffer)

elif input_mode == "Use Microphone":
    st.info("Allow mic access and speak. Audio is processed in chunks.")
    
    # Audio callback to apply noise reduction
    def audio_frame_callback(frame):
        audio = frame.to_ndarray(format="flt32").flatten()
        cleaned = reduce_noise(audio, sr=48000)
        st.session_state.mic_audio = cleaned  # Save latest cleaned audio
        new_frame = av.AudioFrame.from_ndarray(cleaned.astype(np.float32), format="flt32", layout="mono")
        new_frame.sample_rate = 48000
        return new_frame

    # Launch mic stream
    webrtc_streamer(key="mic", audio_frame_callback=audio_frame_callback,
                    media_stream_constraints={"audio": True, "video": False})

    # Show output if we have audio
    if st.session_state.mic_audio is not None:
        st.subheader("Latest Cleaned Mic Audio (Snapshot)")

        cleaned = st.session_state.mic_audio
        sr = 48000

        plot_waveform(cleaned, sr, "Cleaned Mic Audio")

        buffer = BytesIO()
        sf.write(buffer, cleaned, sr, format='wav')
        st.audio(buffer)



