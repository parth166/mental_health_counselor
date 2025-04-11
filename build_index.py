import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load your structured data
with open("data/data.json", "r") as f:
    raw_data = json.load(f)

# Use MiniLM for sentence embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []
metadata = []

for q_id, q in raw_data.items():
    question_text = None
    if q.get("question_text") is not None:
        question_text = q.get("question_text", "").strip()
    if not question_text:
        continue

    texts.append(question_text)

    metadata.append({
        "id": q_id,
        "question": question_text,
        "topic": q["topic"],
        "responses": q.get("responses", [])
    })

# Encode questions
print("🔄 Encoding questions into vectors...")
embeddings = embedding_model.encode(texts, show_progress_bar=True)

# Convert to float32 (required by FAISS)
embeddings = np.array(embeddings).astype("float32")

# Create and index vectors
print("📦 Building FAISS index...")
dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dim)
faiss_index.add(embeddings)

# Save index and metadata
faiss.write_index(faiss_index, "faiss.index")
with open("faiss_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ FAISS index and metadata saved!")
