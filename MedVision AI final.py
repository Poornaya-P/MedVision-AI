import streamlit as st
import google.generativeai as genai
from PIL import Image
import cv2
import tempfile
import time

# ✅ Configure Gemini API
genai.configure(api_key="AIzaSyB4ba31kBRJk8fCQQiNRCbTzkqcgKO-GVE")
model = genai.GenerativeModel("gemini-1.5-flash")

# ✅ Streamlit Page Setup
st.set_page_config(page_title="MedVision AI+", page_icon="🩺", layout="wide")
st.title("🩺 MedVision AI+")
st.markdown("### Unified AI Medical Report Generator")

# Sidebar navigation
st.sidebar.title("🔍 Choose a Service")
service = st.sidebar.radio(
    "Select an option:",
    ["Home", "Blood Test Report Analyzer", "Scan & Imaging Report Generator", "Chat with AI", "Live Endoscopy Analyzer"]
)

if service == "Home":
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=180)
    st.markdown("""
    **MedVision AI+** can:
    - Analyze blood test results
    - Generate structured reports from CT/MRI/X-ray scans
    - Provide chatbot-style clinical Q&A
    - Analyze pre-recorded endoscopy videos for anomalies
    """)
    st.success("Choose a service from the left sidebar to begin.")

elif service == "Blood Test Report Analyzer":
    st.header("🧪 Blood Test Report Analyzer")
    uploaded_file = st.file_uploader("Upload blood report (image/pdf)", type=["png", "jpg", "jpeg", "pdf"])
    if uploaded_file and st.button("Analyze Report"):
        img = Image.open(uploaded_file)
        response = model.generate_content([
            "Analyze this blood report and provide structured findings with flagged abnormal values and a summary for a general physician.",
            img
        ])
        st.success("✅ Report Generated")
        st.write(response.text)

elif service == "Scan & Imaging Report Generator":
    st.header("🩻 Scan & Imaging Report Generator")
    uploaded_scan = st.file_uploader("Upload your X-ray/CT/MRI scan", type=["png", "jpg", "jpeg"])
    if uploaded_scan and st.button("Generate Scan Report"):
        img = Image.open(uploaded_scan)
        response = model.generate_content([
            "Examine this radiology image and generate a structured radiology report (Findings + Impression + Urgency score).",
            img
        ])
        st.success("✅ Report Generated")
        st.write(response.text)

elif service == "Chat with AI":
    st.header("💬 Chat Mode")
    user_input = st.text_input("Ask any health-related question:")
    if st.button("Send"):
        if user_input.strip():
            response = model.generate_content(user_input)
            st.markdown(f"**AI:** {response.text}")
        else:
            st.warning("Please enter a question.")

elif service == "Live Endoscopy Analyzer":
    st.header("🔴 Pre-Recorded Endoscopy Analyzer")
    video_file = st.file_uploader("Upload Endoscopy Video (mp4)", type=["mp4"])

    if video_file:
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(video_file.read())
            video_path = tmp_file.name

        st.video(video_path)
        st.info("Processing video frames every 5 seconds for anomaly detection...")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 5)  # every 5 seconds
        frame_count = 0

        result_box = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Save frame temporarily for AI analysis
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_frame:
                    cv2.imwrite(temp_frame.name, frame)
                    img = Image.open(temp_frame.name)
                    response = model.generate_content([
                        "This is a frame from an endoscopy video. Detect any abnormalities (polyps, lesions, bleeding) and rate severity in 1-5 scale.",
                        img
                    ])
                    result_box.markdown(f"**AI Analysis (Frame {frame_count}):** {response.text}")

            frame_count += 1

        cap.release()
        st.success("✅ Video Analysis Complete")
