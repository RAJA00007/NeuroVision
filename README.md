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



## Overview
NeuroVision is an offline multimodal AI system that integrates **emotion recognition**, **medical image analysis**, and **clinical report summarization** into a unified health-analysis dashboard.  
It demonstrates how computer vision, affective computing, and transformer-based NLP can work together to provide fast, interpretable health insights—while keeping all data completely local and private.

---

## Key Features
- **Real-time emotion detection** from facial expressions  
- **X-ray / MRI / CT analysis** using pretrained Vision Transformer & CNN models  
- **Medical report summarization** using transformer-based NLP  
- **Automated PDF generation** containing all results  
- **Streamlit-powered dashboard** with a clean, unified workflow  
- **100% offline processing** (no API keys, no cloud calls)

---

## System Modules

### 1. Emotion Analysis  
Uses facial-expression recognition to predict emotional states in real time.

### 2. Medical Imaging  
Classifies radiology images and detects anomalies using deep-learning architectures such as Vision Transformers and ResNet.

### 3. Clinical Summarization  
Summarizes long, unstructured medical reports into concise, easy-to-read text.

### 4. Reporting Module  
Generates downloadable PDF health summaries containing:
- Emotion analysis  
- Image classification results  
- Summarized clinical findings  

---

## Technology Stack
- **Python 3.10+**  
- **Streamlit**  
- **DeepFace**, **OpenCV**  
- **Vision Transformers (ViT)**, **ResNet**  
- **HuggingFace Transformers** (BART / T5)  
- **FPDF** (PDF generation)

---

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
