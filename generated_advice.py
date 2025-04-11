import openai
import os
from retrieve_and_classify import build_context_for_query
from openai import OpenAI

client = OpenAI()
openai.api_key = os.getenv("OPENAI_API_KEY")  # Or hardcode for testing

PROMPT_TEMPLATE = """
You are a thoughtful, ethical, and experienced mental health counselor.
Your goal is to assist other counselors and answer their queries.

The counselor is asking:
"{query}"

Below is the context derived from an external source which contains questions related to the query and expert:

{context}

Based on this information answer the counselor's query.

Note: if the counselor is asking information from the external source, keep your response short and concise.
Else: in other cases Keep your tone professional, trauma-informed, and supportive and guide the counselor on how to respond.

Note: Cite relevant advice from the expert responses if available.
"""

def generate_advice(query: str, model="gpt-4o-mini", top_k=20):
    print("🔍 Building context for LLM...")
    results = build_context_for_query(query, top_k_retrieval=top_k)

    prompt = PROMPT_TEMPLATE.format(
        query=results["query"],
        context=results["llm_context"]
    )

    print("🧠 Sending prompt to OpenAI...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=700
    )
    success = True
    try:
        reply = response.choices[0].message.content
    except:
        success = False
    
    return {
        "success": success,
        "generated_response": reply if success else "Sorry, something went wrong! Couldn't generate the advice. Please try again",
        "topic": results["predicted_topic"],
        "context_used": results["llm_context"],
        "top_topics": results["top_topics"]
    }

# Example usage
if __name__ == "__main__":
    test_query = "My client is feeling worthless and withdrawn after losing their job. They have trouble getting out of bed."
    result = generate_advice(test_query)

    print("\n🧾 Suggested Response:\n")
    print(result["generated_response"])

    print("\n🔎 Topic:", result["topic"])
    print("\n📚 Context used (snippet):\n", result["context_used"][:500], "...")
