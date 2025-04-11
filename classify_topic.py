import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import numpy as np

# Load model + tokenizer
MODEL_PATH = "models/bert_topic_model/checkpoint-490"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

def classify_topic(query: str, return_confidence: bool = True):
    # Tokenize the input
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(device)

    # Run model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Get predicted class ID
    pred_id = np.argmax(probs)
    topic = label_encoder.inverse_transform([pred_id])[0]
    
    if return_confidence:
        return {
            "topic": topic,
            "confidence": float(probs[pred_id]),
            "all_probs": dict(zip(label_encoder.classes_, map(float, probs)))
        }
    else:
        return topic

# Example usage
if __name__ == "__main__":
    query = "My client is showing signs of panic attacks and racing thoughts. What’s the best way to approach treatment?"
    result = classify_topic(query)
    print("\n🔍 Predicted Topic:")
    print(f"→ {result['topic']} (confidence: {result['confidence']:.4f})")

    print("\n📊 Full Topic Probabilities:")
    for k, v in sorted(result["all_probs"].items(), key=lambda x: -x[1])[:5]:
        print(f"{k:<20} {v:.3f}")
