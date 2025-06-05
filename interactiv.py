from flask import Flask, request, jsonify
from flask_cors import CORS
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from collections import defaultdict
import numpy as np

app = Flask(__name__)
CORS(app, supports_credentials=True)

MODEL_SAVE_PATH = "model"

LABELS = ['anxiety', 'bipolar', 'stress', 'depression', 'normal', 'personality disorder', 'suicidal']
id_to_label = {i: label for i, label in enumerate(LABELS)}
label_to_id = {label: i for i, label in enumerate(LABELS)}


print(f"Loading fine-tuned model from: {MODEL_SAVE_PATH}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

except Exception as e:
    print(f"Error loading model from {MODEL_SAVE_PATH}: {e}")
    print("Please ensure you have run the fine-tuning script and saved the model correctly.")
    exit()

print("\n--- Mental Health Text Classifier (Non-Clinical Use) ---")
print("Enter text to get a predicted mental health category.")
print("Type 'quit' or 'exit' to stop.")

def preprocess_text(text):
    text = text.lower()
    text = text.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.strip()
    return text

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return {
        'polarity': analysis.sentiment.polarity,
        'subjectivity': analysis.sentiment.subjectivity
    }


def detect_mental_state(text):
    text = preprocess_text(text)
    word_count = defaultdict(int)

    for state, keywords in LABELS.items():
        for keyword in keywords:
            word_count[state] += text.count(keyword)

    if word_count:
        return max(word_count.items(), key=lambda x: x[1])[0]
    return 'normal'

def get_response(category, confidence, sentiment_data, text):
    category = category.lower()

    base_responses = {
        'normal': [
            "I'm glad to hear you're doing well!",
            "That's great to hear!",
            "I'm happy that you're feeling positive."
        ],
        'anxiety': [
            "I understand this might be difficult. Would you like to talk more about what's causing your anxiety?",
            "It's okay to feel anxious sometimes. What specific thoughts or situations are making you feel this way?",
            "I'm here to listen. Would you like to share more about what's troubling you?"
        ],
        'depression': [
            "I'm sorry to hear you're feeling this way. Would you like to talk about what's been on your mind?",
            "It's important to acknowledge these feelings. Would you like to share more about what's been affecting you?",
            "I'm here to support you. Would you like to discuss what's been making you feel this way?"
        ],
        'bipolar': [
            "I notice you're describing some mood changes. Would you like to talk about how you've been feeling lately?",
            "It sounds like you're experiencing some significant mood variations. Would you like to discuss this further?",
            "I'm here to listen. Would you like to share more about your mood patterns?"
        ],
        'stress': [
            "I understand you're feeling stressed. Would you like to talk about what's causing this pressure?",
            "Stress can be overwhelming. What specific situations are making you feel this way?",
            "I'm here to listen. Would you like to share what's been stressing you out?"
        ],
        'personality disorder': [
            "I'm here to listen and support you. Would you like to talk about how you've been feeling?",
            "It sounds like you're going through a difficult time. Would you like to share more?",
            "I'm here to help. Would you like to discuss what's been on your mind?"
        ],
        'suicidal': [
            "I'm concerned about what you're saying. If you're having thoughts of self-harm, please contact a mental health professional immediately. You can reach the National Suicide Prevention Lifeline at 988 or text HOME to 741741 to reach the Crisis Text Line. You're not alone, and help is available."
        ]
    }

    response = np.random.choice(base_responses[category])

    if confidence > 0.8 and category != 'normal' and category != 'suicidal':
        response += " I'm quite confident about this assessment. Would you like to talk more about it?"

    if sentiment_data['polarity'] < -0.5:
        response += " I notice you're feeling quite negative. Would you like to talk about what's bothering you?"
    elif sentiment_data['polarity'] > 0.5:
        response += " I'm glad to see you're feeling positive!"

    if category == 'suicidal':
        response = base_responses['suicidal'][0]

    return response


@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.json
    if not data or 'text' not in data:
        return jsonify({
            'error': 'No text provided in request'
        }), 400

    text = data.get('text', '')
    if not text.strip():
        return jsonify({
            'error': 'Empty text provided'
        }), 400

    sentiment_data = analyze_sentiment(text)

    if classifier is not None:
        try:
            prediction = classifier(text)[0]
            category = prediction['label'].lower()
            confidence = prediction['score']
        except Exception as e:
            print(f"Error in model prediction: {e}")
            print(traceback.format_exc())
            category = detect_mental_state(text)
            confidence = 0.5
    else:
        category = detect_mental_state(text)
        confidence = 0.5

    response = get_response(category, confidence, sentiment_data, text)

    return jsonify({
        'category': category,
        'confidence': confidence,
        'response': response,
        'sentiment': sentiment_data
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)