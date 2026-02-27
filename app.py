import time
import re
import random
import pandas as pd
from flask import Flask, render_template, request, jsonify
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util
from langdetect import detect

# =====================================================
# CONFIG
# =====================================================
MODEL_NAME = "all-MiniLM-L6-v2"
DATASET_PATH = r"C:\Users\HYPE AMD\Medical Bot\dataset - Sheet1.csv"

SIM_LOW = 0.40
SIM_MED = 0.65

app = Flask(__name__)

# =====================================================
# LOAD DATASET
# =====================================================
raw_df = pd.read_csv(DATASET_PATH)
raw_df.columns = raw_df.columns.str.strip().str.lower()

if "disease" not in raw_df.columns or "cure" not in raw_df.columns:
    raise ValueError("Dataset must contain 'disease' and 'cure' columns.")

cleaned_data = []

for _, row in raw_df.iterrows():
    disease_raw = str(row["disease"]).strip()
    cure_text = str(row["cure"]).strip()

    parts = re.split(r"\s{2,}", disease_raw)

    if len(parts) >= 2:
        disease_name = parts[0]
        symptoms = " ".join(parts[1:])
    else:
        words = disease_raw.split()
        disease_name = words[0]
        symptoms = disease_raw

    cleaned_data.append({
        "disease": disease_name,
        "symptoms": symptoms,
        "cure": cure_text
    })

df = pd.DataFrame(cleaned_data)
df = df.drop_duplicates(subset=["disease", "symptoms"])
df.reset_index(drop=True, inplace=True)

# =====================================================
# LOAD MODEL
# =====================================================
model = SentenceTransformer(MODEL_NAME)
symptom_embeddings = model.encode(
    df["symptoms"].tolist(),
    convert_to_tensor=True
)

# =====================================================
# MEDICAL KEYWORD FALLBACK
# =====================================================
medical_keywords = {
    "fever": "It sounds like you may have a fever. Stay hydrated and rest.",
    "cough": "Persistent cough may be due to infection or allergy.",
    "headache": "Headaches can be caused by stress or dehydration.",
    "cold": "Common cold usually resolves within a few days."
}

# =====================================================
# DISEASE INDEX (ICD STYLE MATCH)
# =====================================================
def build_disease_index(df):
    disease_index = {}

    for _, row in df.iterrows():
        name = row["disease"].strip()
        lower = name.lower()
        disease_index[lower] = name
        disease_index[lower.replace(" ", "")] = name
        disease_index[re.sub(r'[^a-z0-9 ]', '', lower)] = name

    alias = {
        "covid": "COVID-19",
        "flu": "Influenza",
        "tb": "Tuberculosis",
        "maag": "Gastritis",
        "gula": "Diabetes",
        "stroke ringan": "Stroke"
    }

    disease_index.update(alias)
    return disease_index


disease_index = build_disease_index(df)

# =====================================================
# UTILITIES
# =====================================================
def summarize_text(text, max_sentences=2):
    sentences = text.split(".")
    summary = ". ".join(sentences[:max_sentences]).strip()
    if not summary.endswith("."):
        summary += "."
    return summary

def detect_language(text):
    try:
        detected = GoogleTranslator(source='auto', target='en').translate(text)
        return GoogleTranslator(source='auto', target='en').source
    except:
        return "en"

def detect_emergency(text):
    emergency_keywords = [
        "severe chest pain",
        "shortness of breath",
        "loss of consciousness",
        "nyeri dada berat",
        "sesak napas"
    ]
    return any(k in text.lower() for k in emergency_keywords)


def extract_keywords(text):
    return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())

# =====================================================
# HARD MATCH
# =====================================================
def keyword_override(user_input):
    words = extract_keywords(user_input)

    for word in words:
        if word in disease_index:
            disease_name = disease_index[word]
            match = df[df["disease"].str.lower() == disease_name.lower()]
            if not match.empty:
                row = match.iloc[0]
                return row["disease"], summarize_text(row["cure"]), 0.97

    return None

# =====================================================
# SEMANTIC MATCH
# =====================================================
def semantic_match(user_input):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, symptom_embeddings)[0]
    best_idx = similarities.argmax().item()
    best_score = similarities[best_idx].item()

    disease = df.iloc[best_idx]["disease"]
    cure = summarize_text(df.iloc[best_idx]["cure"])

    return disease, cure, best_score

# =====================================================
# HEALTH TIPS GENERATOR
# =====================================================
def generate_health_tips(user_input):
    text = user_input.lower()
    tips = []

    if "tired" in text or "fatigue" in text:
        tips.append("Ensure you get enough rest and maintain hydration.")

    if "stress" in text:
        tips.append("Practice relaxation techniques or meditation.")

    if "sleep" in text:
        tips.append("Maintain a consistent sleep schedule.")

    if not tips:
        tips.append("Maintain a healthy lifestyle with proper nutrition.")

    return tips

# =====================================================
# FINAL RESPONSE ENGINE
# =====================================================
def generate_response(user_input):

    if detect_emergency(user_input):
        return (
            "Your symptoms may indicate a serious condition.\n\n"
            "Please seek immediate medical attention."
        )

    override = keyword_override(user_input)

    if override:
        disease, cure, score = override
    else:
        disease, cure, score = semantic_match(user_input)

    tips = generate_health_tips(user_input)

    if score >= SIM_MED:
        intro = f"Based on your symptoms, this may relate to {disease}.\n\n"
    elif score >= SIM_LOW:
        intro = "Your symptoms are somewhat general and may relate to:\n\n"
    else:
        # fallback to keyword medical simple
        for keyword, response in medical_keywords.items():
            if keyword in user_input.lower():
                return response
        return "I don't have enough information. Please consult a healthcare professional."

    tips_block = "\nHelpful suggestions:\n- " + "\n- ".join(tips)

    closing = "\n\nIf symptoms persist or worsen, please consult a doctor."

    return intro + cure + tips_block + closing

# =====================================================
# ROUTES
# =====================================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_input = data.get("message", "").strip()

        if not user_input:
            return jsonify({"reply": "Silakan jelaskan gejala Anda."})

        # 1️⃣ Detect bahasa input user
        try:
            user_lang = detect(user_input)
        except:
            user_lang = "en"

        # 2️⃣ Translate ke English untuk diproses model
        if user_lang != "en":
            translated_input = GoogleTranslator(
                source="auto",
                target="en"
            ).translate(user_input)
        else:
            translated_input = user_input

        # 3️⃣ Generate response dalam English
        response_en = generate_response(translated_input)

        # 4️⃣ Translate kembali ke bahasa user
        if user_lang != "en":
            final_response = GoogleTranslator(
                source="en",
                target=user_lang
            ).translate(response_en)
        else:
            final_response = response_en

        return jsonify({"reply": final_response})

    except Exception as e:
        return jsonify({"reply": "Terjadi kesalahan sistem."})

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)