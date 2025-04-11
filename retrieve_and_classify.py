import torch
import joblib
import faiss
import json
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from topic_classifier_pretrained import get_topics

MODEL_PATH = "models/bert_topic_model/checkpoint-490"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

# topic_tokenizer = AutoTokenizer.from_pretrained("models/bert_topic_model/checkpoint-490")
# topic_model = AutoModelForSequenceClassification.from_pretrained("models/bert_topic_model/checkpoint-490").to("cuda" if torch.cuda.is_available() else "cpu")
# label_encoder = joblib.load("models/label_encoder.pkl")

retriever_model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.read_index("faiss.index")
with open("faiss_metadata.pkl", "rb") as f:
    faiss_metadata = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def classify_topic_custom_model(query):
#     inputs = topic_tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(device)
#     with torch.no_grad():
#         logits = topic_model(**inputs).logits
#         probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
#     pred_id = int(np.argmax(probs))
#     topic = label_encoder.inverse_transform([pred_id])[0]
#     return {
#         "topic": topic,
#         "confidence": float(probs[pred_id]),
#         "all_probs": dict(zip(label_encoder.classes_, map(float, probs)))
#     }

def retrieve_similar_questions(query, top_k=20):
    query_vec = retriever_model.encode([query])
    D, I = faiss_index.search(np.array(query_vec), top_k)
    return [faiss_metadata[i] for i in I[0]]

def filter_by_topic(results, top_topics):
    return [r for r in results if r["topic"] in top_topics]

def rank_responses(questions, top_n_responses=2):
    context_blocks = []
    for q in questions:
        sorted_responses = sorted(q["responses"], key=lambda r: r.get("upvotes", 0), reverse=True)
        for r in sorted_responses[:top_n_responses]:
            context_blocks.append(f"Question: {q['question']}\n\nAnswer: {r['answer_text']}\n\nUpvotes: {r['upvotes']} \n\nCounsellor: {r['therapist_info']}<<end>>")
    return "".join(context_blocks)

# -------- Entry Point -------- #
def build_context_for_query(query, top_k_retrieval=10, top_n_responses=2, top_topic_count=3):
    print("🔍 Running topic classification...")
    # topic_info = classify_topic_custom(query) # we can train a custom model too

    result = get_topics(query)
    top_topics = result['labels'][:top_topic_count]

    print("\n🔎 Retrieving similar questions...")
    retrieved = retrieve_similar_questions(query, top_k=top_k_retrieval)

    print(f"→ Retrieved {len(retrieved)} results. Filtering by topic...")
    filtered = filter_by_topic(retrieved, top_topics)

    print(f"→ {len(filtered)} matched topic-filtered results. Ranking responses...")
    context = rank_responses(filtered, top_n_responses=top_n_responses)

    return {
        "query": query,
        "predicted_topic": top_topics[0],
        "top_topics": top_topics,
        "retrieved_questions": filtered,
        "llm_context": context
    }

# Example
if __name__ == "__main__":
    sample_query = "My client constantly feels worthless and experiences panic attacks after work."
    result = build_context_for_query(sample_query)

    print("\n📚 Final LLM Context:\n")
    print(result["llm_context"][:1000])  # Print the first chunk
