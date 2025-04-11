import streamlit as st
from generated_advice import generate_advice

st.set_page_config(
    page_title="Mental Health Counselor Assistant",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Counselor Assistant")
st.markdown("This assistant helps counselors answer challenging client questions using past expert advice and LLM suggestions.")

# --- Input ---
user_query = st.text_area(
    "📝 Enter a question or case query:",
    placeholder="e.g. My client feels worthless after a breakup and is isolating socially...",
    height=150
)

model_option = st.selectbox("Select OpenAI model:", ["gpt-4o-mini"])
top_k_retrieval = st.slider("How many similar questions to retrieve?", min_value=2, max_value=50, value=10)

# --- Submit ---
if st.button("🧠 Get Advice") and user_query.strip():
    with st.spinner("Analyzing your query, retrieving examples, and generating a response..."):
        result = generate_advice(user_query, model=model_option, top_k=top_k_retrieval)

    # --- Output ---
    st.subheader("🔎 Predicted Topic:")
    st.markdown(f"**{result['topic']}**")

    st.subheader("🗃️ Top Retrieved Questions & Expert Responses")

    context = result["context_used"].split("<<end>>")
    number_steps = len(context)
    

    for idx, item in enumerate(result["context_used"].split("<<end>>")):
        if idx < min(number_steps, top_k_retrieval) - 1:
            st.markdown(f"**{idx + 1}.** {item}")

    st.subheader("🧠 Suggested LLM Response")
    st.markdown(result["generated_response"])

elif user_query.strip() == "":
    st.info("Please enter a question to get started.")

st.markdown("---")
st.markdown("🚀 Built with FAISS, BERT, and OpenAI | By Parth")
