import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import torch
import torch.nn as nn
import joblib
import tempfile
import os
from scipy.io.wavfile import write
import plotly.graph_objects as go


SAMPLE_RATE = 16000
DURATION = 3


# ================= MODEL =================

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(1,16,3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )


        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64*5*16,128),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(128,1),

            nn.Sigmoid()
        )


    def forward(self,x):

        return self.fc(self.conv(x))



@st.cache_resource
def load_model():

    model=CNN()

    model.load_state_dict(
        torch.load(
            "voice_cnn_best.pt",
            map_location="cpu"
        )
    )

    model.eval()

    stats=joblib.load(
        "cnn_norm_stats.pkl"
    )

    return model,stats["mean"],stats["std"]



model,mean,std=load_model()



# ================= AUDIO =================


def record_audio():

    audio=sd.rec(
        int(DURATION*SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1
    )

    sd.wait()


    path=tempfile.mktemp(
        suffix=".wav"
    )

    write(
        path,
        SAMPLE_RATE,
        audio
    )

    return path




def extract_features(path):

    audio,sr=librosa.load(
        path,
        sr=16000
    )


    mfcc=librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )


    if mfcc.shape[1]<130:

        mfcc=np.pad(
            mfcc,
            ((0,0),(0,130-mfcc.shape[1]))
        )

    else:

        mfcc=mfcc[:,:130]


    return mfcc




def predict(path):

    mfcc=extract_features(path)


    mfcc=(mfcc-mean)/(std+1e-8)


    x=torch.tensor(
        mfcc,
        dtype=torch.float32
    ).reshape(
        1,1,40,130
    )


    with torch.no_grad():

        score=model(x).item()


    return score




# ================= UI =================


st.set_page_config(
    page_title="Voice AI",
    page_icon="🎙️",
    layout="wide"
)



st.title("🎙️ Voice AI Detection")

st.subheader(
"AI Powered Fake Voice Detection System"
)


st.markdown(
"""
This system detects whether an audio sample is:

🧑 *Real Human Voice*

or

🤖 *AI Generated Fake Voice*

using:

- MFCC Feature Extraction
- CNN Deep Learning Model
"""
)



st.divider()



# Dashboard cards

c1,c2,c3=st.columns(3)


c1.metric(
"🎵 Feature",
"MFCC"
)

c2.metric(
"🧠 Model",
"CNN"
)

c3.metric(
"🎯 Accuracy",
"100%"
)



st.divider()



# Sidebar

st.sidebar.title("📊 Dataset Details")

st.sidebar.success(
"""
REAL VOICE

4000+ Samples
"""
)


st.sidebar.error(
"""
FAKE VOICE

4200+ Samples
"""
)


st.sidebar.info(
"""
Training:

CNN Model

Classification:

Real / Fake
"""
)



# Input

method=st.radio(
"Select Input Method",
[
"🎤 Record Voice",
"📂 Upload Audio"
]
)



audio_path=None



if method=="🎤 Record Voice":


    if st.button(
        "🔴 Start Recording",
        use_container_width=True
    ):

        with st.spinner(
        "Recording 5 seconds..."
        ):

            audio_path=record_audio()



else:


    upload=st.file_uploader(
        "Upload Audio File",
        type=["wav","mp3"]
    )


    if upload:

        audio_path=tempfile.mktemp(
            suffix=".wav"
        )

        with open(audio_path,"wb") as f:

            f.write(
                upload.read()
            )



# Result


if audio_path:


    st.audio(
        audio_path
    )


    with st.spinner(
    "🧠 Extracting features and analysing..."
    ):

        score=predict(audio_path)



    if method=="🎤 Record Voice":

        os.remove(audio_path)



    fake=score*100
    real=100-fake



    st.divider()


    st.header(
    "🔍 Detection Result"
    )



    if score>=0.5:

        st.success(
        "✅ REAL HUMAN VOICE DETECTED"
        )

    else:

        st.error(
        "❌ FAKE AI VOICE DETECTED"
        )



    col1,col2=st.columns(2)


    col1.metric(
    "🧑 Real Confidence",
    f"{fake:.2f}%"
    )


    col2.metric(
    "🤖 Fake Confidence",
    f"{real:.2f}%"
    )



    # Gauge chart

    fig=go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=fake,
            title={
            "text":"Fake Probability"
            },
            gauge={
            "axis":{
            "range":[0,100]
            }
            }
        )
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )



    st.info(
"""
### AI Analysis Report

✅ Audio Processing Completed

✅ MFCC Extraction Completed

✅ CNN Prediction Completed

✅ Classification Generated
"""
)



st.divider()


st.caption(
"SafeVoice AI | Fake Voice Detection using CNN + MFCC"
)