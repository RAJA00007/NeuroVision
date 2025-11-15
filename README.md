🌟 NeuroVision — AI Health Companion

NeuroVision is a multimodal AI health platform that integrates Emotion Recognition, Medical Image Analysis, and Clinical Report Summarization into one privacy-focused, interactive dashboard.
Built using Python, Streamlit, DeepFace, PyTorch, and Transformer models, the system runs fully offline — ensuring 100% data privacy.

🚀 Key Features

Real-time Emotion Detection (webcam or image)

Medical Image Classification (X-Ray / MRI / CT)

Clinical Report Summarization (BART-based)

PDF Health Report Export

Fully Offline, Privacy-First Inference

Streamlit Dashboard for Unified Experience

🧩 Tech Stack
Category	Tools Used
Frontend	Streamlit UI
Vision	DeepFace, OpenCV, ViT/ResNet
NLP	HuggingFace Transformers (BART/T5)
Backend	Python
Output	FPDF for PDF generation
📥 Installation & Setup
# 1. Clone the repository
git clone https://github.com/RAJA00007/NeuroVision
cd NeuroVision

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py

📂 Project Structure
NeuroVision/
├── app.py                     # Streamlit dashboard
├── requirements.txt
├── assets/                    # banners, diagrams, screenshots
├── models/                    # pretrained models (optional)
├── data/                      # example inputs
├── reports/                   # generated PDFs
├── utils/                     # helper functions
└── README.md

🧠 AI Modules
1. Emotion Detection

Uses DeepFace + OpenCV to extract and classify facial emotions.

from deepface import DeepFace
res = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
print(res[0]['dominant_emotion'])

2. Medical Image Classification

Supports multi-disease classification using ResNet / ViT.

from transformers import AutoImageProcessor, AutoModelForImageClassification
proc = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = proc(img, return_tensors="pt")
logits = model(**inputs).logits

3. Clinical Report Summarization

Summarizes long medical reports into 5–10 line readable text.

from transformers import pipeline
summ = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summ(text, max_length=120, min_length=50)[0]['summary_text']

4. PDF Report Generation

Combines all outputs into a professional PDF.

from fpdf import FPDF
pdf = FPDF()
pdf.add_page()
pdf.multi_cell(0, 10, "Emotion: Happy\nDiagnosis: Normal\nSummary: ...")
pdf.output("report.pdf")

🎛 Streamlit Dashboard

Three-tab streamlined interface:

Emotion Detection

X-ray / MRI Analysis

Report Summarization

All results update live and can be exported.

📊 Evaluation Highlights

Emotion detection accuracy varies with lighting + camera quality

X-ray classifier supports multiple labels (normal, fracture, pneumonia, etc.)

NLP summarizer optimized for medical reports

⚠ Note: This is an educational/assistive tool — not a medical diagnostic system.

🔒 Privacy & Ethical AI

No cloud calls

No external APIs

No data stored permanently

Temporary files auto-deleted

🛠 Contributing

Contributions are welcome!
Feel free to submit:

UI improvements

Additional medical models

Better dataset fine-tuning

Bug fixes

🧾 License

Licensed under the MIT License.

👨‍💻 Author

Raja Kumar Raut
GitHub: @RAJA00007

⭐ Support the Project

If you like this project, feel free to star the repo ⭐ on GitHub!
It motivates more development and improvements.
