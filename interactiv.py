import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os


MODEL_SAVE_PATH = "model/mental_health_model_diag"

LABELS = ['anxiety', 'bipolar', 'stress', 'depression', 'normal', 'personality disorder', 'suicidal']
id_to_label = {i: label for i, label in enumerate(LABELS)}
label_to_id = {label: i for i, label in enumerate(LABELS)}


print(f"Loading fine-tuned model from: {MODEL_SAVE_PATH}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)

except Exception as e:
    print(f"Error loading model from {MODEL_SAVE_PATH}: {e}")
    print("Please ensure you have run the fine-tuning script and saved the model correctly.")
    exit()

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

print("\n--- Mental Health Text Classifier (Non-Clinical Use) ---")
print("Enter text to get a predicted mental health category.")
print("Type 'quit' or 'exit' to stop.")

while True:
    user_input = input("\nEnter your message: ")

    if user_input.lower() in ['quit', 'exit']:
        print("Exiting classifier. Goodbye!")
        break

    if not user_input.strip():
        print("Please enter some text.")
        continue

    try:
        prediction = classifier(user_input)

        predicted_label = prediction[0]['label']
        score = prediction[0]['score']

        print(f"\nPredicted Category: {predicted_label}")
        print(f"Confidence Score: {score:.4f}")

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        print("Please try again or check your model setup.")

print("\n--- Important Disclaimer ---")
print("This model is for informational purposes ONLY and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider with any questions you may have regarding a medical condition or mental health concerns.")