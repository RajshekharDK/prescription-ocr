# 🧠 Prescription OCR — Doctor Handwritten Text Recognition System

An AI-powered web application that extracts and analyzes information from doctor prescriptions using Machine Learning, OCR concepts, and NLP techniques.

---

## 🚀 Project Overview

Prescription OCR digitizes handwritten medical prescriptions by detecting and extracting:

- Patient diagnosis
- Prescribed medicines
- Named medical entities
- Prediction confidence score

This system helps reduce manual errors and supports digital healthcare record management.

---

## ✨ Features

- Handwritten prescription text processing
- Medicine classification
- Diagnosis prediction
- Named Entity Recognition (NER)
- Confidence scoring
- Web-based user interface
- End-to-end ML pipeline

---

## 🛠️ Technologies Used

### Frontend
- React.js
- HTML
- CSS
- JavaScript

### Backend
- Python
- Flask

### Machine Learning
- Scikit-learn
- Pandas
- NumPy
- NLP techniques

---

## 📂 Project Structure

prescription-ocr/
│
├── backend/ # Flask backend API
├── frontend/ # React frontend
├── dataset/ # Training datasets
├── models/ # Trained ML models
├── train_models.py # Model training script
├── rxscan-ai-demo.html # Demo page
└── README.md
---

## ⚙️ Installation & Setup

### 1. Clone Repository


git clone https://github.com/your-username/prescription-ocr.git

cd prescription-ocr


---### 2. Backend Setup


cd backend
pip install -r requirements.txt
python app.py


Backend will run at: http://localhost:5000

---

### 3. Frontend Setup


cd frontend
npm install
npm start


Frontend will run at: http://localhost:3000

---

## 🧪 Model Training

To retrain the models:


python train_models.py


---

## 📊 Dataset

The dataset includes labeled prescription data such as medicine names, diagnosis labels, and extracted entities.

---

## 🎯 Use Cases

- Hospitals and clinics
- Pharmacies
- Medical record digitization
- Healthcare AI research

---

## 🔮 Future Improvements

- Deep learning OCR integration
- Real-time camera scanning
- Multi-language support
- Cloud deployment
- Mobile application

---

## 👨‍💻 Author

Rajshekhar DK  
Computer Science Engineering Student

---

## 📜 License

This project is for educational and research purposes only.
