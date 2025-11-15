<div align="center">
<img src=ChatGPT Image Nov 16, 2025, 02_29_55 AM (2).png>
<h2>NeuroVision — AI Health Companion</h2>
<p>Multimodal AI for Emotion Analysis • Medical Imaging • Clinical Summaries</p>
<!-- Badges -->
<img src="https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square">
<img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square">
<img src="https://img.shields.io/badge/Platform-Offline%20AI-darkblue?style=flat-square">
<img src="https://img.shields.io/badge/License-MIT-black?style=flat-square">
</div>


Overview

NeuroVision is an offline multimodal AI system that integrates emotion recognition, medical image analysis, and clinical report summarization into a unified health-analysis dashboard.
It demonstrates how computer vision, affective computing, and transformer-based NLP can work together to provide rapid, interpretable insights while maintaining full data privacy.

Key Features

Real-time facial emotion recognition

X-ray/MRI/CT image classification using pretrained deep-learning models

Summarization of medical reports using transformer architectures

Automatic PDF generation containing all results

Unified Streamlit dashboard for seamless interaction

Fully offline execution with no API calls or cloud dependency

System Modules

Emotion Analysis
Uses facial-expression recognition models to predict the user’s emotional state in real time.

Medical Imaging
Processes uploaded radiology images to classify potential abnormalities using Vision Transformers or CNN-based architectures.

Clinical Summarization
Converts long, unstructured medical reports into concise summaries for easier interpretation.

Reporting Module
Generates downloadable PDF health summaries combining outputs from all modules.

Technology Stack

Python (3.10+)

Streamlit

DeepFace, OpenCV

Vision Transformers / ResNet

HuggingFace Transformers (BART/T5)

FPDF

Installation
pip install -r requirements.txt
streamlit run app.py

Folder Structure
NeuroVision/
│── app.py
│── requirements.txt
│── assets/
│── data/
│── reports/
│── utils/
└── README.md

Privacy & Ethics

All computation takes place locally.
No user data is uploaded, logged, or transmitted.
This project is intended purely for academic, research, and demonstration purposes — not for clinical use.

Roadmap

Explainable AI (attention maps, Grad-CAM)

Support for additional imaging modalities

Optimized models for mobile devices

On-device small-LLM integration

UI workflow enhancements

Author

Raja Kumar Raut
GitHub: https://github.com/RAJA00007
