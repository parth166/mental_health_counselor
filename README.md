# 🧠 Mental Health Counselor Assistant

A powerful, AI-powered assistant to help mental health counselors respond to complex client situations using insights drawn from expert therapist responses and large language models (LLMs). Built with a Retrieval-Augmented Generation (RAG) pipeline, local ML models, and OpenAI.

Run it on: https://counsellor.streamlit.app/

---

## 💡 What It Does

This web application helps mental health counselors:

- 💬 **Ask free-text queries** about client cases
- 🔍 **Retrieve similar expert responses** from a curated mental health counseling corpus
- 🧠 **Generate high-quality suggestions** via OpenAI or local LLM
- 🏷️ **Automatically classify queries** into mental health topics (e.g., anxiety, trauma, grief)

---

## 🧱 Project Structure

mental-health-assistant/ 
├── app.py # Streamlit frontend 
├── retrieve_and_classify.py # Semantic search + topic classification 
├── generate_llm_advice.py # LLM response generator 
├── train_topic_classifier.py# Fine-tunes a DistilBERT/Roberta classifier 
├── data/ 
│ ├── data.json # Full mental health Q&A dataset 
│ ├── grouped_by_topic.json
│ ├── therapists.json
├── faiss.index # Dense vector search index 
├── faiss_metadata.pkl # Metadata mapped to FAISS vectors 
├── models/ 
│ └── bert_topic_model/ # Fine-tuned classification model 
│ └── label_encoder.pkl # Sklearn label encoder 
├── requirements.txt 
└── README.md

---

## 📊 Features & Deliverables

### ✅ Application Deliverables

- **Dataset**: Expert mental health Q&A dataset (from CounselChat)
- **Topic Classifier**: 
  - Fine-tuned BERT model trained on question → topic mappings
  - Leveraged a BART model based NLI model as a classifer and leveraged topic models for query classification
- **LLM RAG Pipeline**:
  - Semantic search using FAISS + SentenceTransformer
  - Optional topic filtering + upvote-based response ranking
  - LLM prompt construction + GPT-4o-mini response

---

## 🧪 How to Run Locally

```bash
# Clone repo
git clone https://github.com/yourusername/mental-health-assistant.git
cd mental-health-assistant

# Set up environment
pip install -r requirements.txt

# Set your OpenAI key
export OPENAI_API_KEY="sk-..."

# Run the app
streamlit run app.py

---