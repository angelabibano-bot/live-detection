import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Detection", layout="wide")

# ---------- CUSTOM UI STYLE ----------
st.markdown("""
<style>
/* Background gradient */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Title */
.title {
    font-size: 40px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 30px;
}

/* Card */
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 20px rgba(0,0,0,0.3);
}

/* Slider label */
.css-1cpxqw2 {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title">🎥 AI Object Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time detection using YOLOv8 + Webcam</div>', unsafe_allow_html=True)

# ---------- LAYOUT ----------
col1, col2 = st.columns([3, 1])

# ---------- SETTINGS PANEL ----------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚙️ Controls")

    confidence = st.slider("Confidence", 0.1, 0.9, 0.25)
    frame_skip = st.slider("Smoothness (Frame Skip)", 1, 5, 3)

    st.markdown("### ℹ️ Tips")
    st.write("• Higher skip = smoother video")
    st.write("• Lower confidence = more detections")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- MODEL ----------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------- PROCESSOR ----------
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
            boxes = results[0].boxes
            names = model.names

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = names[cls_id]
                conf = float(box.conf[0])

                # box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)

                # label
                cv2.putText(
                    annotated,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ---------- VIDEO AREA ----------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    webrtc_streamer(
        key="live-detect",
        video_processor_factory=VideoProcessor,
        async_processing=True,
        media_stream_constraints={
            "video": {
                "width": 480,
                "height": 360,
                "frameRate": 15
            },
            "audio": False,
        },
    )

    st.markdown('</div>', unsafe_allow_html=True)