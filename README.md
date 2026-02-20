🧠 Prescription OCR — Doctor Handwritten Text Recognition System

An AI-powered web application that extracts, analyzes, and predicts information from doctor prescriptions using Machine Learning, OCR, and NLP.

🚀 Project Overview

Prescription OCR is designed to digitize handwritten medical prescriptions by detecting and extracting:

🩺 Patient diagnosis

💊 Medicines

📊 Confidence score

🧾 Structured medical data

This system helps reduce manual errors, improves record keeping, and assists in digital healthcare workflows.

🏗️ Project Architecture
Frontend (React)  →  Backend (Flask API)  →  ML Models  →  Prediction Output
✨ Features

✔ Handwritten prescription text extraction
✔ Medicine name classification
✔ Diagnosis prediction
✔ Named Entity Recognition (NER)
✔ Confidence scoring
✔ User-friendly web interface
✔ End-to-end ML pipeline

🛠️ Technologies Used
🔹 Frontend

React.js

HTML

CSS

JavaScript

🔹 Backend

Python

Flask

🔹 Machine Learning

Scikit-learn

Pandas

NumPy

NLP techniques

🔹 Models Included

Medicine Classifier

Diagnosis Predictor

NER Extractor

Confidence Scorer

📂 Project Structure
prescription-ocr/
│
├── backend/                # Flask API
├── frontend/               # React App
├── dataset/                # Training datasets
├── models/                 # Trained ML models
├── train_models.py         # Model training script
├── rxscan-ai-demo.html     # Demo page
└── README.md
⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/your-username/prescription-ocr.git
cd prescription-ocr
2️⃣ Backend Setup
cd backend
pip install -r requirements.txt
python app.py

Backend runs at:

http://localhost:5000
3️⃣ Frontend Setup
cd frontend
npm install
npm start

Frontend runs at:

http://localhost:3000
🧪 Model Training

To retrain models:

python train_models.py
📊 Dataset

The dataset contains labeled prescription data including:

Medicine names

Diagnosis labels

Extracted entities

🎯 Use Cases

🏥 Hospitals
💊 Pharmacies
🧑‍⚕️ Doctors
📋 Medical record digitization
🧠 Healthcare AI research

🔮 Future Improvements

Deep Learning OCR integration

Real-time camera scanning

Multi-language prescription support

Cloud deployment

Mobile application

👨‍💻 Author

Rajshekhar DK
Computer Science Engineering Student
AI & ML Enthusiast

📜 License

This project is for educational and research purposes.
