"""
Doctor Prescription OCR - Flask Backend
========================================
Uses Tesseract OCR + ML for handwritten text recognition
"""


'''
from flask import Flask, request, jsonify
from flask_cors import CORS
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import pandas as pd
import re
import os
import base64
import io
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────
# Load Dataset for Context / Medicine DB
# ─────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), '../dataset/prescription_dataset.csv')
try:
    df = pd.read_csv(DATASET_PATH)
    MEDICINE_DB = df['medicine_name'].str.lower().unique().tolist()
    print(f"[✓] Loaded {len(MEDICINE_DB)} medicines from dataset")
except Exception as e:
    print(f"[!] Dataset not loaded: {e}")
    MEDICINE_DB = []

# ─────────────────────────────────────────
# Frequency Mapping
# ─────────────────────────────────────────
FREQUENCY_MAP = {
    'od': 'Once Daily', 'once daily': 'Once Daily',
    'bd': 'Twice Daily', 'bid': 'Twice Daily', 'twice daily': 'Twice Daily',
    'tds': 'Thrice Daily', 'tid': 'Thrice Daily', 'thrice daily': 'Thrice Daily',
    'qid': 'Four Times Daily', 'four times': 'Four Times Daily',
    'hs': 'At Bedtime', 'bedtime': 'At Bedtime', 'night': 'At Bedtime',
    'sos': 'As Needed', 'prn': 'As Needed', 'as needed': 'As Needed',
    'stat': 'Immediately', 'immediately': 'Immediately',
    'weekly': 'Once Weekly', 'once weekly': 'Once Weekly',
    'morning': 'Every Morning', 'am': 'Every Morning',
}

# ─────────────────────────────────────────
# Image Preprocessing
# ─────────────────────────────────────────
def preprocess_image(image: Image.Image) -> Image.Image:
    """Enhance image for better OCR accuracy"""
    # Convert to numpy array
    img_array = np.array(image.convert('RGB'))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Adaptive thresholding for handwritten text
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Dilation to thicken strokes (helps with faint handwriting)
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    # Convert back to PIL
    processed = Image.fromarray(dilated)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(processed)
    processed = enhancer.enhance(2.0)
    
    # Sharpen
    processed = processed.filter(ImageFilter.SHARPEN)
    
    return processed


# ─────────────────────────────────────────
# OCR Engine
# ─────────────────────────────────────────
def run_ocr(image: Image.Image) -> dict:
    """Run Tesseract OCR with multiple configs for best result"""
    preprocessed = preprocess_image(image)
    
    configs = [
        '--oem 3 --psm 6',   # Assume a single uniform block of text
        '--oem 3 --psm 4',   # Assume a single column
        '--oem 3 --psm 11',  # Sparse text
    ]
    
    best_text = ''
    best_conf = 0
    
    for cfg in configs:
        try:
            data = pytesseract.image_to_data(
                preprocessed, config=cfg,
                output_type=pytesseract.Output.DICT
            )
            conf_vals = [c for c in data['conf'] if c > 0]
            avg_conf = sum(conf_vals) / len(conf_vals) if conf_vals else 0
            
            if avg_conf > best_conf:
                best_conf = avg_conf
                best_text = pytesseract.image_to_string(preprocessed, config=cfg)
        except Exception:
            pass
    
    return {
        'text': best_text.strip(),
        'confidence': round(best_conf / 100, 2)
    }


# ─────────────────────────────────────────
# NLP Parser for Prescription Fields
# ─────────────────────────────────────────
def parse_prescription(text: str) -> dict:
    """Extract structured data from OCR text using regex + NLP"""
    text_lower = text.lower()
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    # --- Doctor Name ---
    doctor_pattern = re.search(r'dr\.?\s+([a-z\s\.]+)', text_lower)
    doctor_name = doctor_pattern.group(0).title().strip() if doctor_pattern else 'Unknown'
    
    # --- Patient Name ---
    patient_patterns = [
        r'patient\s*:?\s*([a-z\s]+)',
        r'name\s*:?\s*([a-z\s]+)',
        r'pt\.?\s*:?\s*([a-z\s]+)',
    ]
    patient_name = 'Unknown'
    for pat in patient_patterns:
        m = re.search(pat, text_lower)
        if m:
            patient_name = m.group(1).strip().title()
            break
    
    # --- Date ---
    date_pattern = re.search(
        r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})',
        text
    )
    date_val = date_pattern.group(0) if date_pattern else datetime.now().strftime('%Y-%m-%d')
    
    # --- Medicines ---
    medicines = extract_medicines(text, text_lower)
    
    # --- Diagnosis ---
    diag_patterns = [r'diagnosis\s*:?\s*([^\n]+)', r'dx\s*:?\s*([^\n]+)', r'c/o\s*:?\s*([^\n]+)']
    diagnosis = ''
    for pat in diag_patterns:
        m = re.search(pat, text_lower)
        if m:
            diagnosis = m.group(1).strip().title()
            break
    
    return {
        'doctor_name': doctor_name,
        'patient_name': patient_name,
        'date': date_val,
        'diagnosis': diagnosis,
        'medicines': medicines,
        'raw_text': text,
    }


def extract_medicines(text: str, text_lower: str) -> list:
    """Extract medicine entries from text"""
    results = []
    
    # Patterns for Tab/Cap/Syp/Inj + medicine name + dose
    med_pattern = re.findall(
        r'(tab(?:let)?|cap(?:sule)?|syr(?:up)?|syp|inj(?:ection)?|drop|oint(?:ment)?|neb)\.?\s+'
        r'([a-z][a-z\s\+\-]+?)\s+'
        r'(\d+(?:\.\d+)?(?:mg|mcg|ml|g|iu|%|units?)?(?:\/\d+(?:ml|g)?)?)',
        text_lower
    )
    
    for match in med_pattern:
        form, name, dose = match
        name = name.strip()
        
        # Find frequency
        freq = ''
        for abbr, full in FREQUENCY_MAP.items():
            if abbr in text_lower:
                freq = full
                break
        
        # Find duration
        dur_match = re.search(r'x\s*(\d+)\s*(day|week|month)s?', text_lower)
        duration = dur_match.group(0).replace('x ', '') if dur_match else 'Ongoing'
        
        # Find instructions
        instructions = ''
        for kw in ['after food', 'before food', 'with food', 'empty stomach', 'at bedtime', 'with meals']:
            if kw in text_lower:
                instructions = kw.title()
                break
        
        results.append({
            'form': form.title(),
            'name': name.title(),
            'dosage': dose,
            'frequency': freq or 'As Directed',
            'duration': duration,
            'instructions': instructions,
        })
    
    # Fallback: scan for known medicines in DB
    if not results:
        for med in MEDICINE_DB:
            if med in text_lower:
                results.append({
                    'form': 'Tablet',
                    'name': med.title(),
                    'dosage': 'As Prescribed',
                    'frequency': 'As Directed',
                    'duration': 'As Directed',
                    'instructions': '',
                })
    
    return results


# ─────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'medicines_in_db': len(MEDICINE_DB)})


@app.route('/api/ocr', methods=['POST'])
def ocr_endpoint():
    """Main OCR endpoint - accepts image file or base64"""
    try:
        if 'image' in request.files:
            file = request.files['image']
            image = Image.open(file.stream)
        elif request.is_json and 'image_base64' in request.json:
            b64 = request.json['image_base64']
            # Strip data URL prefix if present
            if ',' in b64:
                b64 = b64.split(',')[1]
            img_bytes = base64.b64decode(b64)
            image = Image.open(io.BytesIO(img_bytes))
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Run OCR
        ocr_result = run_ocr(image)
        
        # Parse prescription
        parsed = parse_prescription(ocr_result['text'])
        parsed['ocr_confidence'] = ocr_result['confidence']
        
        return jsonify({
            'success': True,
            'data': parsed,
            'confidence': ocr_result['confidence']
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/dataset', methods=['GET'])
def get_dataset():
    """Return sample dataset records"""
    try:
        records = df.head(20).to_dict(orient='records')
        return jsonify({'success': True, 'data': records, 'total': len(df)})
    except Exception:
        return jsonify({'success': False, 'data': [], 'total': 0})


@app.route('/api/medicines', methods=['GET'])
def get_medicines():
    """Return medicine list from DB"""
    query = request.args.get('q', '').lower()
    if query:
        filtered = [m for m in MEDICINE_DB if query in m]
    else:
        filtered = MEDICINE_DB
    return jsonify({'medicines': filtered[:50]})


@app.route('/api/demo', methods=['GET'])
def demo_result():
    """Return a demo parsed prescription for frontend testing"""
    return jsonify({
        'success': True,
        'confidence': 0.89,
        'data': {
            'doctor_name': 'Dr. R. Sharma',
            'patient_name': 'Ravi Kumar',
            'date': '2024-01-10',
            'diagnosis': 'Bacterial Infection',
            'raw_text': 'Dr. R. Sharma\nPatient: Ravi Kumar\nDt: 10/01/2024\nDx: Bacterial Infection\n\nTab Amoxicillin 500mg BD x 5 days - After meals\nTab Paracetamol 500mg SOS - As needed for fever\nSyp Benadryl 2 tsp TDS x 5 days',
            'medicines': [
                {'form': 'Tab', 'name': 'Amoxicillin', 'dosage': '500mg', 'frequency': 'Twice Daily', 'duration': '5 days', 'instructions': 'After Meals'},
                {'form': 'Tab', 'name': 'Paracetamol', 'dosage': '500mg', 'frequency': 'As Needed', 'duration': '5 days', 'instructions': 'For Fever'},
                {'form': 'Syp', 'name': 'Benadryl', 'dosage': '2 tsp', 'frequency': 'Thrice Daily', 'duration': '5 days', 'instructions': ''},
            ],
            'ocr_confidence': 0.89
        }
    })


if __name__ == '__main__':
    print("🏥 Prescription OCR Server starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)

'''

# update code

"""
RxScan AI — Flask Backend with 4 ML Models
============================================
Models used:
  1. medicine_classifier.pkl     → predicts medicine name from OCR text
  2. diagnosis_predictor.pkl     → predicts diagnosis from OCR text
  3. ner_extractor.pkl           → extracts entities (medicine/dosage/freq/form)
  4. confidence_scorer.pkl       → scores OCR confidence (0.0–1.0)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os, re, pickle, json, base64, io
from datetime import datetime

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    import cv2
    import numpy as np
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

import pandas as pd

app = Flask(__name__)
CORS(app)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../models")
DATASET    = os.path.join(BASE_DIR, "../dataset/prescription_dataset_100k.csv")

def load_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    print(f"[!] Model not found: {filename}")
    return None

print("[*] Loading ML models...")
med_classifier = load_model("medicine_classifier.pkl")
med_encoder    = load_model("medicine_label_encoder.pkl")
diag_predictor = load_model("diagnosis_predictor.pkl")
diag_encoder   = load_model("diagnosis_label_encoder.pkl")
ner_model      = load_model("ner_extractor.pkl")
conf_package   = load_model("confidence_scorer.pkl")

model_info_path = os.path.join(MODELS_DIR, "model_info.json")
model_info = json.load(open(model_info_path)) if os.path.exists(model_info_path) else {}

try:
    df = pd.read_csv(DATASET, nrows=10000)
    MEDICINE_DB = df["medicine_name"].str.lower().unique().tolist()
    print(f"[✓] Loaded {len(MEDICINE_DB)} medicines")
except Exception as e:
    MEDICINE_DB = []
print(f"[✓] Models ready | OCR: {OCR_AVAILABLE}")

def extract_confidence_features(text):
    text_lower = text.lower()
    return [
        len(text), len(text.split()), len(text.split("|")),
        int(bool(re.search(r"dr\.?\s+\w+", text_lower))),
        int(bool(re.search(r"patient\s*:", text_lower))),
        int(bool(re.search(r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}", text))),
        int(bool(re.search(r"\d+\s*mg", text_lower))),
        int(bool(re.search(r"\b(od|bd|tds|qid|sos)\b", text_lower))),
        int(bool(re.search(r"\brx\b", text_lower))),
        int(bool(re.search(r"diagnosis|dx", text_lower))),
        int(any(m in text_lower for m in MEDICINE_DB[:50])),
        text.count("\n"),
        len(re.findall(r"\d+\s*mg", text_lower)),
        len(re.findall(r"\b(tab|cap|syp|inj)\b", text_lower)),
        int(bool(re.search(r"signature", text_lower))),
        len(set(text.lower().split())) / max(len(text.split()), 1),
        sum(c.isupper() for c in text) / max(len(text), 1),
        sum(c.isdigit() for c in text) / max(len(text), 1),
        int(bool(re.search(r"advice|instructions", text_lower))),
        int(bool(re.search(r"\b(mg|ml|mcg|iu|units?)\b", text_lower))),
    ]

def predict_medicine(text):
    if not med_classifier or not med_encoder:
        return None
    try:
        pred       = med_classifier.predict([text])[0]
        proba      = med_classifier.predict_proba([text])[0]
        confidence = float(max(proba))
        medicine   = med_encoder.inverse_transform([pred])[0]
        return {"medicine": medicine, "confidence": round(confidence, 3)}
    except:
        return None

def predict_diagnosis(text):
    if not diag_predictor or not diag_encoder:
        return None
    try:
        pred       = diag_predictor.predict([text])[0]
        proba      = diag_predictor.predict_proba([text])[0]
        confidence = float(max(proba))
        diagnosis  = diag_encoder.inverse_transform([pred])[0]
        return {"diagnosis": diagnosis, "confidence": round(confidence, 3)}
    except:
        return None

def predict_confidence(text):
    if not conf_package:
        return 0.85
    try:
        import numpy as np
        model   = conf_package["model"]
        scaler  = conf_package["scaler"]
        feats   = extract_confidence_features(text)
        X       = np.array([feats])
        score   = float(model.predict(scaler.transform(X))[0])
        return round(min(max(score, 0.0), 1.0), 3)
    except:
        return 0.85

def ner_extract(text):
    if not ner_model:
        return {}
    text_lower = text.lower()
    patterns   = ner_model.get("patterns", {})
    freq_map   = ner_model.get("freq_map", {})
    result     = {}

    m = re.search(patterns.get("doctor", ""), text_lower)
    result["doctor_name"] = m.group(0).title().strip() if m else ""

    for pat in [r"patient\s*:?\s*([a-z][a-z\s]{2,25})", r"pt\.?\s*:?\s*([a-z][a-z\s]{2,20})"]:
        m = re.search(pat, text_lower)
        if m:
            result["patient_name"] = m.group(1).strip().title()
            break
    result.setdefault("patient_name", "")

    m = re.search(patterns.get("date", ""), text)
    result["date"] = m.group(0) if m else datetime.now().strftime("%Y-%m-%d")

    for pat in [r"diagnosis\s*:?\s*([^\n\|]{3,40})", r"dx\s*:?\s*([^\n\|]{3,40})"]:
        m = re.search(pat, text_lower)
        if m:
            result["diagnosis"] = m.group(1).strip().title()
            break
    result.setdefault("diagnosis", "")

    med_pattern = re.findall(
        r"(tab(?:let)?s?|cap(?:sule)?s?|syr(?:up)?|syp|inj(?:ection)?|drop|oint(?:ment)?|neb)\.?\s+"
        r"([a-z][a-z\s\+\-]+?)\s+"
        r"(\d+(?:\.\d+)?(?:mg|mcg|ml|g|iu|%|units?)(?:\/\d+(?:ml|g)?)?)",
        text_lower
    )

    medicines = []
    for form, name, dose in med_pattern:
        freq_abbr = ""
        for abbr in ["qid", "tds", "tid", "bid", "bd", "od", "sos", "hs", "weekly"]:
            if re.search(r"\b" + abbr + r"\b", text_lower):
                freq_abbr = abbr.upper()
                break
        dur_m    = re.search(r"x\s*(\d+\s*(?:day|week|month)s?)", text_lower)
        duration = dur_m.group(1) if dur_m else "Ongoing"
        instr    = ""
        for kw in ["after food", "before food", "with food", "empty stomach", "after breakfast", "with meals"]:
            if kw in text_lower:
                instr = kw.title()
                break
        medicines.append({
            "form": form.title(), "name": name.strip().title(), "dosage": dose,
            "frequency": freq_map.get(freq_abbr.lower(), freq_abbr or "As Directed"),
            "frequency_abbr": freq_abbr, "duration": duration, "instructions": instr,
        })
    result["medicines"] = medicines
    return result

def preprocess_image(image):
    if not OCR_AVAILABLE:
        return image
    try:
        import numpy as np
        img_array = np.array(image.convert("RGB"))
        gray      = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        denoised  = cv2.fastNlMeansDenoising(gray, h=10)
        binary    = cv2.adaptiveThreshold(denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed = Image.fromarray(binary)
        processed = ImageEnhance.Contrast(processed).enhance(2.0)
        return processed.filter(ImageFilter.SHARPEN)
    except:
        return image

def run_ocr(image):
    if not OCR_AVAILABLE:
        return {"text": "", "confidence": 0.5}
    try:
        processed = preprocess_image(image)
        best_text, best_conf = "", 0
        for cfg in ["--oem 3 --psm 6", "--oem 3 --psm 4", "--oem 3 --psm 11"]:
            data  = pytesseract.image_to_data(processed, config=cfg, output_type=pytesseract.Output.DICT)
            confs = [c for c in data["conf"] if c > 0]
            avg   = sum(confs) / len(confs) if confs else 0
            if avg > best_conf:
                best_conf = avg
                best_text = pytesseract.image_to_string(processed, config=cfg)
        return {"text": best_text.strip(), "confidence": round(best_conf / 100, 2)}
    except Exception as e:
        return {"text": "", "confidence": 0.5}

# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "ocr_available": OCR_AVAILABLE,
        "models_loaded": {
            "medicine_classifier": med_classifier is not None,
            "diagnosis_predictor": diag_predictor is not None,
            "ner_extractor": ner_model is not None,
            "confidence_scorer": conf_package is not None,
        },
        "medicines_in_db": len(MEDICINE_DB),
    })

@app.route("/api/ocr", methods=["POST"])
def ocr_endpoint():
    try:
        if "image" in request.files:
            image = Image.open(request.files["image"].stream)
        elif request.is_json and "image_base64" in request.json:
            b64 = request.json["image_base64"]
            if "," in b64: b64 = b64.split(",")[1]
            image = Image.open(io.BytesIO(base64.b64decode(b64)))
        else:
            return jsonify({"error": "No image provided"}), 400

        ocr_result  = run_ocr(image)
        text        = ocr_result["text"]
        entities    = ner_extract(text)
        ml_conf     = predict_confidence(text)
        ml_medicine = predict_medicine(text) if text else None
        ml_diag     = predict_diagnosis(text) if text else None

        if not entities.get("diagnosis") and ml_diag:
            entities["diagnosis"] = ml_diag["diagnosis"]
        if not entities.get("medicines") and ml_medicine:
            entities["medicines"] = [{"form":"Tab","name":ml_medicine["medicine"],"dosage":"As Prescribed","frequency":"As Directed","frequency_abbr":"","duration":"As Directed","instructions":""}]

        return jsonify({"success": True, "confidence": ml_conf,
            "data": {**entities, "raw_text": text, "ocr_confidence": ml_conf,
                "ml_predictions": {"medicine": ml_medicine, "diagnosis": ml_diag}}})
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route("/api/predict/medicine", methods=["POST"])
def predict_medicine_api():
    text = request.json.get("text", "") if request.is_json else ""
    return jsonify({"success": True, "prediction": predict_medicine(text)})

@app.route("/api/predict/diagnosis", methods=["POST"])
def predict_diagnosis_api():
    text = request.json.get("text", "") if request.is_json else ""
    return jsonify({"success": True, "prediction": predict_diagnosis(text)})

@app.route("/api/predict/confidence", methods=["POST"])
def predict_confidence_api():
    text = request.json.get("text", "") if request.is_json else ""
    return jsonify({"success": True, "confidence": predict_confidence(text)})

@app.route("/api/ner", methods=["POST"])
def ner_api():
    text = request.json.get("text", "") if request.is_json else ""
    return jsonify({"success": True, "entities": ner_extract(text)})

@app.route("/api/models", methods=["GET"])
def models_info():
    return jsonify({"success": True, "model_info": model_info,
        "loaded": {
            "medicine_classifier": med_classifier is not None,
            "diagnosis_predictor": diag_predictor is not None,
            "ner_extractor": ner_model is not None,
            "confidence_scorer": conf_package is not None,
        }})

@app.route("/api/dataset", methods=["GET"])
def get_dataset():
    try:
        return jsonify({"success": True, "data": df.head(20).to_dict(orient="records"), "total": len(df)})
    except:
        return jsonify({"success": False, "data": [], "total": 0})

@app.route("/api/demo", methods=["GET"])
def demo():
    sample_text = """Dr. R. Sharma  MD, MBBS
Patient: Ravi Kumar    Date: 10/01/2024
Diagnosis: Bacterial Infection

Rx
Tab Amoxicillin 500mg BD x 5 days - After food
Tab Paracetamol 500mg SOS - For fever
Syp Benadryl 5ml TDS x 5 days - After food
Cap Vitamin C 500mg OD - After breakfast

Advice: Rest, plenty of fluids.
Signature: Dr. R. Sharma"""

    entities    = ner_extract(sample_text)
    ml_conf     = predict_confidence(sample_text)
    ml_medicine = predict_medicine(sample_text)
    ml_diag     = predict_diagnosis(sample_text)

    if not entities.get("medicines"):
        entities["medicines"] = [
            {"form":"Tab","name":"Amoxicillin","dosage":"500mg","frequency":"Twice Daily","frequency_abbr":"BD","duration":"5 days","instructions":"After Food"},
            {"form":"Tab","name":"Paracetamol","dosage":"500mg","frequency":"As Needed","frequency_abbr":"SOS","duration":"5 days","instructions":"For Fever"},
            {"form":"Syp","name":"Benadryl","dosage":"5ml","frequency":"Thrice Daily","frequency_abbr":"TDS","duration":"5 days","instructions":"After Food"},
            {"form":"Cap","name":"Vitamin C","dosage":"500mg","frequency":"Once Daily","frequency_abbr":"OD","duration":"1 month","instructions":"After Breakfast"},
        ]

    return jsonify({"success": True, "confidence": ml_conf, "data": {
        **entities,
        "doctor_name": entities.get("doctor_name") or "Dr. R. Sharma",
        "patient_name": entities.get("patient_name") or "Ravi Kumar",
        "date": entities.get("date") or "2024-01-10",
        "diagnosis": entities.get("diagnosis") or (ml_diag["diagnosis"] if ml_diag else "Bacterial Infection"),
        "raw_text": sample_text, "ocr_confidence": ml_conf,
        "ml_predictions": {"medicine": ml_medicine, "diagnosis": ml_diag},
    }})

if __name__ == "__main__":
    print("\n🏥 RxScan AI — ML-Powered Prescription OCR")
    print(f"   Models dir : {MODELS_DIR}")
    print(f"   OCR engine : {'Tesseract' if OCR_AVAILABLE else 'Not installed'}")
    print("   Server     : http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)



