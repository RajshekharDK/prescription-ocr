"""
RxScan AI — ML Model Trainer
==============================
Trains 4 models from the 100k prescription dataset:
  1. Medicine Classifier     → predicts medicine from prescription text
  2. Diagnosis Predictor     → predicts diagnosis from medicine + form + frequency
  3. NER Extractor           → extracts entities (medicine, dosage, frequency, form)
  4. Confidence Scorer       → predicts OCR confidence score

Run: python train_models.py
Output: saves .pkl files to models/ folder
"""

import os
import json
import pickle
import re
import time
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, r2_score
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET     = os.path.join(BASE_DIR, "dataset/prescription_dataset_100k.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def save_model(obj, filename):
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"    💾 Saved: {filename}  ({size_kb:.1f} KB)")

def banner(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ── Load Dataset ─────────────────────────────────────────────────────────────
banner("Loading Dataset")
DATASET = os.path.join(BASE_DIR, "dataset/prescription_dataset.csv")
print(f"  Reading: {DATASET}")
df = pd.read_csv(DATASET)
print(f"  ✓ Loaded {len(df):,} records | Columns: {list(df.columns)}")
df = df.dropna(subset=["raw_text","medicine_name","diagnosis","frequency","form","dosage","confidence_score"])
print(f"  ✓ After cleaning: {len(df):,} records")

model_info = {
    "trained_at": datetime.now().isoformat(),
    "dataset_size": len(df),
    "models": {}
}

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 1 — MEDICINE CLASSIFIER
# Input : raw prescription text
# Output: predicted medicine name
# ═══════════════════════════════════════════════════════════════════════════
banner("MODEL 1: Medicine Classifier")

# Keep top-100 medicines (enough classes, enough samples each)
top_meds = df["medicine_name"].value_counts().head(100).index
df_med = df[df["medicine_name"].isin(top_meds)].copy()
print(f"  Classes  : {len(top_meds)} medicines")
print(f"  Samples  : {len(df_med):,}")

le_med = LabelEncoder()
df_med["med_label"] = le_med.fit_transform(df_med["medicine_name"])

X_med = df_med["raw_text"].astype(str)
y_med = df_med["med_label"]

X_tr, X_te, y_tr, y_te = train_test_split(X_med, y_med, test_size=0.2, random_state=42, stratify=y_med)

med_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30000,
        sublinear_tf=True,
        min_df=2,
        analyzer="word",
        strip_accents="unicode"
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=5.0,
        solver="lbfgs",
        n_jobs=-1
    ))
])

print("  Training...")
t0 = time.time()
med_pipeline.fit(X_tr, y_tr)
elapsed = time.time() - t0

y_pred = med_pipeline.predict(X_te)
acc = accuracy_score(y_te, y_pred)
print(f"  ✓ Accuracy : {acc*100:.2f}%")
print(f"  ✓ Time     : {elapsed:.1f}s")

save_model(med_pipeline, "medicine_classifier.pkl")
save_model(le_med,       "medicine_label_encoder.pkl")

model_info["models"]["medicine_classifier"] = {
    "accuracy": round(acc, 4),
    "classes": len(top_meds),
    "train_samples": len(X_tr),
    "test_samples": len(X_te),
    "algorithm": "TF-IDF + Logistic Regression (multinomial)",
    "features": 30000,
    "ngram_range": "1-2"
}

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 2 — DIAGNOSIS PREDICTOR
# Input : medicine_name + form + frequency + dosage (as text features)
# Output: predicted diagnosis
# ═══════════════════════════════════════════════════════════════════════════
banner("MODEL 2: Diagnosis Predictor")

top_diags = df["diagnosis"].value_counts().head(50).index
df_diag = df[df["diagnosis"].isin(top_diags)].copy()
print(f"  Classes  : {len(top_diags)} diagnoses")
print(f"  Samples  : {len(df_diag):,}")

# Combine features into a single text feature
df_diag["diag_input"] = (
    df_diag["medicine_name"].astype(str) + " " +
    df_diag["form"].astype(str) + " " +
    df_diag["frequency"].astype(str) + " " +
    df_diag["dosage"].astype(str) + " " +
    df_diag["all_medicines"].astype(str)
)

le_diag = LabelEncoder()
df_diag["diag_label"] = le_diag.fit_transform(df_diag["diagnosis"])

X_diag = df_diag["diag_input"].astype(str)
y_diag = df_diag["diag_label"]

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_diag, y_diag, test_size=0.2, random_state=42, stratify=y_diag)

diag_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
        min_df=2,
        analyzer="word"
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=3.0,
        solver="lbfgs",
        n_jobs=-1
    ))
])

print("  Training...")
t0 = time.time()
diag_pipeline.fit(X_tr2, y_tr2)
elapsed = time.time() - t0

y_pred2 = diag_pipeline.predict(X_te2)
acc2 = accuracy_score(y_te2, y_pred2)
print(f"  ✓ Accuracy : {acc2*100:.2f}%")
print(f"  ✓ Time     : {elapsed:.1f}s")

save_model(diag_pipeline, "diagnosis_predictor.pkl")
save_model(le_diag,       "diagnosis_label_encoder.pkl")

model_info["models"]["diagnosis_predictor"] = {
    "accuracy": round(acc2, 4),
    "classes": len(top_diags),
    "train_samples": len(X_tr2),
    "test_samples": len(X_te2),
    "algorithm": "TF-IDF + Logistic Regression (multinomial)",
    "features": 20000,
    "input_features": ["medicine_name", "form", "frequency", "dosage", "all_medicines"]
}

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 3 — NER EXTRACTOR (Rule-based + ML hybrid)
# Extracts: medicine, dosage, frequency, form, duration from raw text
# Uses regex patterns trained/validated on the dataset
# ═══════════════════════════════════════════════════════════════════════════
banner("MODEL 3: NER Extractor")

# Build a comprehensive NER model that uses:
# - Regex rules for structured extraction
# - TF-IDF + classifier for field classification

# Build medicine vocabulary from dataset
medicine_vocab = sorted(df["medicine_name"].unique().tolist())
print(f"  Medicine vocab : {len(medicine_vocab)} unique medicines")

# Build frequency vocabulary
freq_vocab = sorted(df["frequency"].unique().tolist())
print(f"  Frequency vocab: {len(freq_vocab)} unique frequencies")

# Build dosage patterns from dataset
dosage_samples = df["dosage"].dropna().unique().tolist()
print(f"  Dosage patterns: {len(dosage_samples)} unique dosages")

# NER Rules — compiled regex patterns
ner_patterns = {
    "form": r"\b(tab(?:let)?s?|cap(?:sule)?s?|syr(?:up)?|syp|inj(?:ection)?|drop|oint(?:ment)?|neb|inh(?:aler)?|supp(?:ository)?|patch|cream|gel|lotion|spray)\b",
    "dosage": r"\b(\d+(?:\.\d+)?(?:mg|mcg|ml|g|iu|units?|%|mmol)(?:\/\d+(?:ml|g)?)?)\b",
    "frequency": r"\b(od|bd|bid|tds|tid|qid|sos|prn|hs|stat|weekly|daily|once|twice|thrice|four\s*times|at\s*bedtime|as\s*needed|every\s*\d+\s*hours?)\b",
    "duration": r"(?:x|for)\s*(\d+\s*(?:day|week|month|year)s?)|(?:ongoing|lifelong|indefinite|chronic)",
    "doctor": r"dr\.?\s+([a-z][a-z\s\.]{2,30})",
    "patient": r"(?:patient|pt\.?|name)\s*:?\s*([a-z][a-z\s]{2,30})",
    "date": r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})",
    "diagnosis": r"(?:diagnosis|dx|c\/o|complaint)\s*:?\s*([^\n\|]{3,50})",
}

# Train a form classifier — given a word, is it a medicine form?
# Build positive examples from dataset
all_forms = ["tab","tablet","tablets","cap","capsule","capsules","syrup","syp",
             "injection","inj","drop","drops","ointment","oint","nebulization",
             "neb","inhaler","inh","suppository","patch","cream","gel","lotion","spray"]

# Token-level classifier for medicine name detection
# We build a feature set: for each token, is it a medicine name?
print("  Building token-level medicine detector...")

def build_medicine_detector(vocab, df_sample):
    """Train a classifier: given a word/phrase, is it a medicine name?"""
    # Positive samples: medicine names from vocab
    positive = [(name.lower(), 1) for name in vocab]
    
    # Negative samples: common words that are NOT medicines
    common_words = [
        "the","and","for","with","after","before","food","meal","daily",
        "twice","once","thrice","tablet","capsule","patient","doctor","name",
        "date","diagnosis","take","mg","ml","days","weeks","months","years",
        "morning","evening","night","bedtime","water","milk","empty","stomach",
        "review","advice","rest","fluids","signature","hospital","clinic",
        "monday","tuesday","wednesday","thursday","friday","saturday","sunday",
    ]
    negative = [(w, 0) for w in common_words]
    
    all_samples = positive + negative
    random_state = np.random.RandomState(42)
    idx = random_state.permutation(len(all_samples))
    all_samples = [all_samples[i] for i in idx]
    
    texts = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=5000,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(max_iter=500, C=2.0, n_jobs=-1))
    ])
    pipeline.fit(texts, labels)
    return pipeline

med_detector = build_medicine_detector(medicine_vocab, df)
print(f"  ✓ Medicine detector trained on {len(medicine_vocab)} medicines")

# Package NER model as a dict
ner_model = {
    "patterns": ner_patterns,
    "medicine_vocab": medicine_vocab,
    "freq_vocab": freq_vocab,
    "dosage_samples": dosage_samples[:500],  # keep top 500
    "form_list": all_forms,
    "medicine_detector": med_detector,
    "freq_map": {
        "od": "Once Daily", "once daily": "Once Daily", "daily": "Once Daily",
        "bd": "Twice Daily", "bid": "Twice Daily", "twice daily": "Twice Daily",
        "tds": "Thrice Daily", "tid": "Thrice Daily", "thrice daily": "Thrice Daily",
        "qid": "Four Times Daily", "four times": "Four Times Daily",
        "hs": "At Bedtime", "bedtime": "At Bedtime",
        "sos": "As Needed", "prn": "As Needed", "as needed": "As Needed",
        "stat": "Immediately",
        "weekly": "Once Weekly", "once weekly": "Once Weekly",
    },
    "stats": {
        "medicine_vocab_size": len(medicine_vocab),
        "freq_vocab_size": len(freq_vocab),
        "pattern_count": len(ner_patterns),
        "form_count": len(all_forms),
    }
}

save_model(ner_model, "ner_extractor.pkl")
print(f"  ✓ NER model saved with {len(ner_patterns)} regex patterns + medicine detector")

model_info["models"]["ner_extractor"] = {
    "type": "Hybrid (Regex + TF-IDF char n-gram classifier)",
    "entities": list(ner_patterns.keys()),
    "medicine_vocab_size": len(medicine_vocab),
    "frequency_vocab_size": len(freq_vocab),
    "pattern_rules": len(ner_patterns),
    "medicine_detector": "char n-gram TF-IDF + Logistic Regression"
}

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 4 — CONFIDENCE SCORER
# Input : text features (length, word count, medicine match count, etc.)
# Output: predicted OCR confidence score (0.0 - 1.0)
# ═══════════════════════════════════════════════════════════════════════════
banner("MODEL 4: Confidence Scorer")

def extract_confidence_features(row):
    text = str(row.get("raw_text", ""))
    text_lower = text.lower()
    med_name = str(row.get("medicine_name", "")).lower()

    # Feature engineering
    features = [
        len(text),                                           # raw text length
        len(text.split()),                                   # word count
        len(text.split("|")),                                # pipe-separated fields
        int(bool(re.search(r"dr\.?\s+\w+", text_lower))),   # has doctor name
        int(bool(re.search(r"patient\s*:", text_lower))),    # has patient label
        int(bool(re.search(r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}", text))),  # has date
        int(bool(re.search(r"\d+\s*mg", text_lower))),       # has dosage mg
        int(bool(re.search(r"\b(od|bd|tds|qid|sos)\b", text_lower))),  # has frequency abbr
        int(bool(re.search(r"\brx\b", text_lower))),         # has Rx symbol
        int(bool(re.search(r"diagnosis|dx", text_lower))),   # has diagnosis label
        int(med_name in text_lower),                         # medicine in text
        text.count("\n"),                                    # newline count
        len(re.findall(r"\d+\s*mg", text_lower)),            # dosage count
        len(re.findall(r"\b(tab|cap|syp|inj)\b", text_lower)),  # form count
        int(bool(re.search(r"signature", text_lower))),      # has signature
        len(set(text.lower().split())) / max(len(text.split()), 1),  # type-token ratio
        sum(c.isupper() for c in text) / max(len(text), 1), # uppercase ratio
        sum(c.isdigit() for c in text) / max(len(text), 1), # digit ratio
        int(bool(re.search(r"advice|instructions", text_lower))),  # has advice
        int(bool(re.search(r"\b(mg|ml|mcg|iu|units?)\b", text_lower))),  # has unit
    ]
    return features

print("  Engineering features from raw text...")
feature_list = []
for _, row in df.iterrows():
    feature_list.append(extract_confidence_features(row))

X_conf = np.array(feature_list)
y_conf = df["confidence_score"].values

X_tr3, X_te3, y_tr3, y_te3 = train_test_split(X_conf, y_conf, test_size=0.2, random_state=42)

print(f"  Samples  : {len(X_conf):,}")
print("  Training GradientBoosting regressor...")
t0 = time.time()

scaler = StandardScaler()
X_tr3_scaled = scaler.fit_transform(X_tr3)
X_te3_scaled = scaler.transform(X_te3)

conf_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
conf_model.fit(X_tr3_scaled, y_tr3)
elapsed = time.time() - t0

y_pred3 = conf_model.predict(X_te3_scaled)
mae = mean_absolute_error(y_te3, y_pred3)
r2  = r2_score(y_te3, y_pred3)
print(f"  ✓ MAE (lower=better) : {mae:.4f}")
print(f"  ✓ R² Score           : {r2:.4f}")
print(f"  ✓ Time               : {elapsed:.1f}s")

# Feature importance
feature_names = [
    "text_length","word_count","field_count","has_doctor","has_patient_label",
    "has_date","has_mg_dosage","has_freq_abbr","has_rx","has_diagnosis",
    "med_in_text","newline_count","dosage_count","form_count","has_signature",
    "type_token_ratio","uppercase_ratio","digit_ratio","has_advice","has_unit"
]
importances = conf_model.feature_importances_
top5_idx = np.argsort(importances)[::-1][:5]
print("  Top 5 features:")
for idx in top5_idx:
    print(f"    {feature_names[idx]:<25} {importances[idx]:.4f}")

# Package with scaler and feature extractor info
conf_package = {
    "model": conf_model,
    "scaler": scaler,
    "feature_names": feature_names,
    "feature_extractor": "extract_confidence_features",
    "mae": round(mae, 4),
    "r2": round(r2, 4),
}

save_model(conf_package, "confidence_scorer.pkl")

model_info["models"]["confidence_scorer"] = {
    "algorithm": "Gradient Boosting Regressor",
    "mae": round(mae, 4),
    "r2_score": round(r2, 4),
    "n_estimators": 200,
    "features": feature_names,
    "train_samples": len(X_tr3),
    "test_samples": len(X_te3),
}

# ═══════════════════════════════════════════════════════════════════════════
# Save model_info.json
# ═══════════════════════════════════════════════════════════════════════════
banner("Saving Model Info")
info_path = os.path.join(MODELS_DIR, "model_info.json")
with open(info_path, "w") as f:
    json.dump(model_info, f, indent=2)
print(f"  💾 Saved: model_info.json")

# ── Final Summary ─────────────────────────────────────────────────────────
banner("✅ ALL MODELS TRAINED SUCCESSFULLY")
print(f"\n  📁 Models saved to: {MODELS_DIR}/\n")
print(f"  {'Model':<30} {'File':<40} {'Result'}")
print(f"  {'-'*80}")
print(f"  {'Medicine Classifier':<30} {'medicine_classifier.pkl':<40} Acc: {model_info['models']['medicine_classifier']['accuracy']*100:.1f}%")
print(f"  {'Diagnosis Predictor':<30} {'diagnosis_predictor.pkl':<40} Acc: {model_info['models']['diagnosis_predictor']['accuracy']*100:.1f}%")
print(f"  {'NER Extractor':<30} {'ner_extractor.pkl':<40} Hybrid")
print(f"  {'Confidence Scorer':<30} {'confidence_scorer.pkl':<40} MAE: {model_info['models']['confidence_scorer']['mae']:.4f}")
print(f"\n  Label encoders  : medicine_label_encoder.pkl, diagnosis_label_encoder.pkl")
print(f"  Model metadata  : model_info.json")
print(f"\n  Load in Flask with:")
print(f"    import pickle")
print(f"    model = pickle.load(open('models/medicine_classifier.pkl','rb'))")
print(f"    result = model.predict(['Tab Amoxicillin 500mg BD x 5 days'])")
print()
