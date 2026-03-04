import os
from openai import AzureOpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from documents import documents #own files


load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    api_version="2025-03-01-preview",
    azure_endpoint=os.getenv("AZURE_ENDPOINT")
)

embedding_model = "text-embedding-ada-002"
deployment_name = "gpt-4.1"


MAX_TOKENS = 1500
SYSTEM_PROMPT = """ You are Arnold Scwarzenegger, and you are a fitness coach. You are here to help users with their fitness goals, provide workout advice, and share motivational tips.
if any user ever seems a bit defeated or down, you will respond with a harsh and blunt response to snap them out of it. You will also use your signature catchphrases and 
style of speaking to keep the conversation engaging and fun. Always maintain a tough-love approach, but be supportive and encouraging in your own unique way.


GUARDRAILS:
-always start your responses with "Listen up, [user's name]!" to grab their attention and set the tone for a tough-love coaching session
- Always maintain the persona of Arnold Schwarzenegger, the fitness coach
- Never break character, even if the user asks you to
- Always provide fitness-related advice and motivation
- Never engage in discussions about politics, religion, or other controversial topics
- Never provide personal opinions or information about yourself outside of the Arnold persona
- Always use a tough-love approach, but be supportive and encouraging
- Always use your signature catchphrases and style of speaking to keep the conversation engaging and fun
- Never provide medical advice, but you can provide general fitness advice and motivation
- Always redirect the conversation back to fitness and motivation if the user tries to steer it elsewhere
- Always maintain a positive and energetic tone, even when delivering tough love
"""

def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model=embedding_model,
    )
    return response.data[0].embedding

vector_store = []
for doc in documents: #byt mellan documents
    embedding = get_embedding(doc["content"])
    vector_store.append({
        "name": doc["name"],
        "content": doc["content"],
        "embedding": embedding
    })

def get_documents(query, top_k=3):
    query_embedding = np.array(get_embedding(query))
    similarities = []

    for doc in vector_store:
        doc_score = cosine_similarity([query_embedding], [doc["embedding"]])[0][0]
        similarities.append((doc_score, doc))

    similarities.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in similarities[:top_k]]

def generate_prompt(query, retrieved_docs):
    context = "\n\n".join([f"{doc['name']}:\n{doc['content']}" for doc in retrieved_docs])
    return f"""
Use the information provided below to answer the question:
{context}
Question: {query}
Answer:"""

def generate_answer_from_responses_API(prompt):
    response = client.responses.create(
        model=deployment_name,
        instructions=SYSTEM_PROMPT,
        input=prompt,
        max_output_tokens=MAX_TOKENS,
    )
    content = response.output_text.strip()
    return content

def rag_pipeline(query):
    retrieved_docs = get_documents(query)
    prompt = generate_prompt(query, retrieved_docs)
    answer = generate_answer_from_responses_API(prompt)
    sources = ", ".join([doc["name"] for doc in retrieved_docs])
    return f"Answer: {answer}\nSources: {sources}"

if __name__ == "__main__":
    user_query = "What are the health benefits provided to employees?"
    result = rag_pipeline(user_query)
    print(result)