# =========================================================
# 🧠 NeuroVision 8.7 — AI Health Companion (Final Build)
# Emotion + AI Health Chat + Smart X-Ray + Report Summarizer + PDF Generator
# =========================================================

# ---- SAFE IMPORTS (IMPORTANT FOR STREAMLIT CLOUD) ----
import os, time, threading, json, wave

# Safe cv2 import
try:
    import cv2
except Exception:
    cv2 = None

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from statistics import mode
from fpdf import FPDF

# ---------- SAFE FOLDERS ----------
for path in ["data", "data/session_logs", "reports"]:
    os.makedirs(path, exist_ok=True)

# ---------- UI ----------
st.set_page_config(page_title="NeuroVision 8.7", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
  background:radial-gradient(circle at 10% 20%,#111827 0%,#0f172a 80%);
  color:#e5e7eb;
}
h1,h2,h3{color:#93c5fd;}
div.stButton>button,.stDownloadButton button{
  background:linear-gradient(90deg,#6366f1,#3b82f6);
  color:white;border:0;border-radius:8px;padding:.4rem 1rem;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 NeuroVision — AI Clinical Health Companion")

# ---------- STATE ----------
if "log" not in st.session_state:
    st.session_state.log = []

if "sid" not in st.session_state:
    st.session_state.sid = f"session_{datetime.now():%Y%m%d_%H%M%S}"

# ---------- SIDEBAR ----------
st.sidebar.header("Controls")
patient = st.sidebar.text_input("Patient Name", "")
smooth = st.sidebar.slider("Smoothing (frames)", 3, 10, 5)
talk_conf = st.sidebar.slider("Confidence Threshold", 0.5, 0.9, 0.6, 0.01)
run = st.sidebar.toggle("▶️ Run Live Session", False)

# ---------- CHATBOT ----------
def get_chat(msg: str) -> str:
    rules = {
        "happy": "Patient appears positive and stable.",
        "sad": "Signs of sadness detected — mild mood disturbance likely.",
        "fear": "Anxiety cues observed. Suggest relaxation and breathing exercises.",
        "angry": "Stress and irritation levels high. Recommend calm breathing.",
        "neutral": "Emotional state appears balanced and stable.",
        "surprise": "Alert state detected, no major stress response.",
        "disgust": "Negative emotional cues — may need stress intervention."
    }
    for k, v in rules.items():
        if k in msg.lower():
            return v
    return "Stable emotional state detected."

def speak_async(txt):
    def _run():
        try:
            import pyttsx3
            e = pyttsx3.init()
            e.setProperty('rate', 165)
            e.say(txt)
            e.runAndWait()
            e.stop()
        except:
            pass
    threading.Thread(target=_run, daemon=True).start()

# ---------- TABS ----------
tab_main, tab_chat, tab_scan, tab_pdf = st.tabs([
    "🧭 Emotion",
    "💬 AI Health Chat",
    "🩻 X-Ray & Report Summary",
    "📋 Clinical Report"
])

# =========================================================
# 🧭 EMOTION MONITOR
# =========================================================
if run:

    if cv2 is None:
        tab_main.error("❌ Camera is not supported on Streamlit Cloud.")
    else:
        with st.spinner("Loading DeepFace model …"):
            from deepface import DeepFace
            import plotly.graph_objects as go

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Camera not accessible.")
            st.stop()

        st.success("Camera started.")
        frame_ph, bar_ph = tab_main.empty(), tab_main.empty()
        last = None
        stable = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                res = DeepFace.analyze(rgb, actions=['emotion'], enforce_detection=False)[0]
                emo = res['dominant_emotion'].capitalize()
                conf = res['emotion'][res['dominant_emotion']] / 100
            except Exception:
                emo, conf = "Neutral", 0.5
                res = {'emotion': {'neutral': 100}}

            st.session_state.log.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "emotion": emo,
                "conf": conf
            })

            if len(st.session_state.log) > 400:
                st.session_state.log.pop(0)

            hist = [e['emotion'] for e in st.session_state.log[-smooth:]]

            try:
                smoothed = mode(hist)
            except:
                smoothed = emo

            cv2.putText(frame, f"{smoothed} ({conf:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (80, 220, 120), 2)

            frame_ph.image(rgb)

            # BAR CHART
            emotions = res['emotion']
            fig = go.Figure([go.Bar(x=list(emotions.keys()), y=list(emotions.values()))])
            fig.update_yaxes(range=[0, 100])
            bar_ph.plotly_chart(fig, use_container_width=True, key=f"emotion_chart_{time.time()}")

            # Auto Report Speech
            if smoothed == last:
                stable += 1
            else:
                last, stable = smoothed, 1

            if stable >= 3 and conf >= talk_conf:
                reply = get_chat(smoothed)
                speak_async(reply)
                tab_main.info(f"🩺 AI Report: {reply}")
                stable = 0

            time.sleep(.25)

            if not run:
                break

        cap.release()
        cv2.destroyAllWindows()

# =========================================================
# 💬 AI HEALTH CHATBOT
# =========================================================
with tab_chat:
    st.subheader("💬 AI Health Chat Assistant")

    st.markdown("""
    Ask about your reports, scans, or emotional results.  
    Examples:  
    • What does my X-ray summary mean?  
    • How can I lower my stress level?  
    • What is my emotional wellbeing score?
    """)

    user_query = st.text_area("🩺 Ask a question:", placeholder="Type your medical or emotional question...")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Build context safely
    context_parts = []

    for name in ("result", "short_summary"):
        val = globals().get(name) or locals().get(name)
        if val:
            context_parts.append(val)

    if len(st.session_state.log) > 0:
        last_em = st.session_state.log[-1]["emotion"]
        context_parts.append(f"Detected emotion: {last_em}")

    context = " ".join(context_parts) or "General patient health context."

    if st.button("Ask AI"):
        if not user_query.strip():
            st.warning("Please type a question.")
        else:
            with st.spinner("Analyzing..."):
                from transformers import pipeline
                try:
                    qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
                    answer = qa(question=user_query, context=context)["answer"]
                    if not answer:
                        answer = "I need more context or a report."
                except Exception as e:
                    answer = f"⚠️ Model error: {e}"

            st.session_state.chat_history.append(("🧍‍♂️ You", user_query))
            st.session_state.chat_history.append(("🤖 AI", answer))

    if st.session_state.chat_history:
        st.markdown("### 💭 Conversation History")
        for role, msg in st.session_state.chat_history[-12:]:
            color = "#1e293b" if role == "🤖 AI" else "transparent"
            st.markdown(
                f"<div style='background:{color};padding:8px;border-radius:8px;margin-bottom:5px;'>"
                f"**{role}:** {msg}</div>", unsafe_allow_html=True
            )

# =========================================================
# 🩻 X-RAY / REPORT ANALYZER
# =========================================================
with tab_scan:
    st.subheader("🩻 Smart X-Ray / MRI / CT Analyzer + Report Summary")

    from transformers import AutoImageProcessor, AutoModelForImageClassification
    import torch
    from PIL import Image

    def detect_body_part(img):
        w, h = img.size
        r = h / w
        if r > 1.3:
            return "chest"
        elif r < 0.8:
            return "bone"
        return "brain"

    def get_model_id(part):
        return {
            "chest": "prithivMLmods/chest-disease-classification",
            "bone": "Heem2/bone-fracture-detection-using-xray",
            "brain": "Heem2/brain-tumor-classification"
        }.get(part)

    img_up = st.file_uploader("📸 Upload any medical scan", type=["jpg", "jpeg", "png"])

    if img_up:
        img = Image.open(img_up).convert("RGB")
        st.image(img, width=350)
        part = detect_body_part(img)
        MODEL_ID = get_model_id(part)
        st.info(f"Detected scan type → **{part.upper()}**")

        try:
            with st.spinner("Running diagnostics..."):
                proc = AutoImageProcessor.from_pretrained(MODEL_ID)
                model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
                inputs = proc(img, return_tensors="pt")
                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = logits.softmax(1).cpu().numpy()[0]
                    label = model.config.id2label[int(probs.argmax())]
                    conf = float(probs.max())

            st.success(f"Prediction: **{label}** ({conf:.1%})")

        except Exception as e:
            st.error(f"Model load error: {e}")

    # REPORT SUMMARY
    st.markdown("---")
    st.subheader("📄 Quick Medical Report Summary")

    report_file = st.file_uploader("📤 Upload Doctor’s Report (PDF / Image)",
                                   type=["pdf", "jpg", "jpeg", "png"])

    if report_file:
        text_data = ""

        if report_file.name.endswith(".pdf"):
            import fitz
            doc = fitz.open(stream=report_file.read(), filetype="pdf")
            for p in doc:
                text_data += p.get_text("text")
        else:
            import pytesseract
            img = Image.open(report_file)
            text_data = pytesseract.image_to_string(img)

        text_data = " ".join(text_data.split())

        if len(text_data) > 3000:
            text_data = text_data[:3000] + "..."

        if text_data.strip():
            st.success("Report text extracted!")

            with st.expander("Extracted Text"):
                st.text_area("Report Text", text_data, height=250)

            try:
                from transformers import pipeline
                summarizer = pipeline("summarization",
                                      model="facebook/bart-large-cnn")
                result = summarizer(text_data, max_length=80,
                                    min_length=25, do_sample=False)[0]["summary_text"]
                st.success(result)

            except:
                lines = text_data.split(".")[:3]
                short_summary = "\n".join(f"• {ln.strip()}" for ln in lines)
                st.info(short_summary)

# =========================================================
# 📋 CLINICAL REPORT GENERATION
# =========================================================
with tab_pdf:
    st.subheader("📋 Generate Clinical Emotion Report")

    if st.button("Generate PDF Report"):
        df = pd.DataFrame(st.session_state.log)

        if df.empty:
            st.warning("Run an emotion session first.")
            st.stop()

        dom = mode(df["emotion"])
        avg = df["conf"].mean()
        stress = (1 - avg) * 100
        relax = avg * 100
        well = (avg * 0.6 + 0.5 * 0.4) * 100
        dur = len(df) * 0.25 / 60

        report = f"""
Patient: {patient or 'N/A'}
Session Duration: {dur:.1f} mins
Dominant Emotion: {dom}
Confidence: {avg:.2f}
Stress Index: {stress:.1f}%
Relax Score: {relax:.1f}%
Wellbeing Score: {well:.1f}%

Observation:
{get_chat(dom)}

Disclaimer: This AI report is NOT a medical diagnosis.
"""

        pdf_path = f"reports/Report_{st.session_state.sid}.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 15)
        pdf.cell(0, 10, "NeuroVision Report", 0, 1, "C")
        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(180, 8, report)
        pdf.output(pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Report",
                               data=f,
                               file_name=os.path.basename(pdf_path),
                               mime="application/pdf")
