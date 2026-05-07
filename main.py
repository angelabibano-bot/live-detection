import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
from twilio.rest import Client
import av
import cv2

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Live Object Detection & Tracing", layout="wide")

# ---------- UI DESIGN ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #38bdf8;
    margin-top: 10px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 20px;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎥Live Object Detection & Tracing </div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">YOLOv8 + Webcam + Streamlit WebRTC</div>', unsafe_allow_html=True)

# ---------- LAYOUT ----------
col1, col2 = st.columns([3, 1])

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    confidence = st.slider("Confidence", 0.1, 0.9, 0.3)
    frame_skip = st.slider("Frame Skip", 1, 5, 3)

    st.write("⚡ Lower skip = smoother detection")
    st.write("🎯 Lower confidence = more detections")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- MODEL ----------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------- TWILIO SETUP ----------
account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
auth_token = st.secrets["TWILIO_AUTH_TOKEN"]

client = Client(account_sid, auth_token)
token = client.tokens.create()

# ---------- VIDEO PROCESSOR ----------
class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.frame_count = 0
        self.last_result = None

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (480, 360))

        self.frame_count += 1

        if self.frame_count % frame_skip == 0:
            results = model.predict(img, conf=confidence, verbose=False)
            self.last_result = results
        else:
            results = self.last_result

        annotated = img.copy()

        if results and results[0].boxes is not None:

            names = model.names

            for box in results[0].boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])

                label = names[cls_id]

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)

                cv2.putText(
                    annotated,
                    f"{label} {conf_val:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ---------- LIVE CAMERA ----------
with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    webrtc_streamer(
        key="live-detect",

        video_processor_factory=VideoProcessor,

        async_processing=True,

        rtc_configuration={
            "iceServers": token.ice_servers
        },

        media_stream_constraints={
            "video": True,
            "audio": False
        },
    )

    st.markdown('</div>', unsafe_allow_html=True)
