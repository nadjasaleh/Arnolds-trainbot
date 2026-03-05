from langchain_text_splitters import RecursiveCharacterTextSplitter
from azure.ai.contentunderstanding import ContentUnderstandingClient
from azure.ai.contentunderstanding.models import AnalysisInput
from azure.core.credentials import AzureKeyCredential
from azure.cosmos import CosmosClient, PartitionKey
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

endpoint = os.getenv("CONTENT_UNDERSTANDING_ENDPOINT")
key = os.getenv("CONTENT_UNDERSTANDING_PRIMARY_KEY")
if not endpoint or not key:
    raise RuntimeError("CONTENT_UNDERSTANDING_ENDPOINT or CONTENT_UNDERSTANDING_PRIMARY_KEY missing (.env)")

client = ContentUnderstandingClient(endpoint=endpoint, credential=AzureKeyCredential(key))

cosmos_endpoint = os.getenv("COSMOS_ENDPOINT_arnord")
cosmos_key = os.getenv("COSMOS_KEY_arnord")
if not cosmos_endpoint or not cosmos_key:
    raise RuntimeError("COSMOS_ENDPOINT_arnord or COSMOS_KEY_arnord missing (.env)")

cosmos_client = CosmosClient(cosmos_endpoint, cosmos_key)

COSMOS_DB_ID = os.getenv("COSMOS_DB_ID", "AI_Database")
COSMOS_CONTAINER_ID = os.getenv("COSMOS_CONTAINER_ID", "EmbeddingsContainer")
database = cosmos_client.get_database_client(COSMOS_DB_ID)
container = database.get_container_client(COSMOS_CONTAINER_ID)

CHAT_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT")
if not CHAT_MODEL:
    raise RuntimeError("AZURE_OPENAI_DEPLOYMENT missing (.env)")

EMBED_MODEL = "text-embedding-ada-002"


def get_embedding(text: str) -> list[float]:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def retrieve_top_k(query: str, k: int = 3) -> list[dict]:
    query_embedding = get_embedding(query)

    cosmos_query = f"""
        SELECT TOP {k}
            c.id,
            c.text,
            VectorDistance(c.embedding, @query_vector) AS score
        FROM c
        ORDER BY VectorDistance(c.embedding, @query_vector)
    """

    parameters = [{"name": "@query_vector", "value": query_embedding}]

    items = container.query_items(
        query=cosmos_query,
        parameters=parameters,
        enable_cross_partition_query=True
    )

    results = []
    for item in items:
        results.append({
            "score": float(item.get("score", 0.0)),
            "id": item.get("id", "cosmos-item"),
            "text": item.get("text", "")
        })

    return results


def build_grounded_prompt(user_query: str, retrieved_docs: list[dict]) -> list[dict]:
    # Include chunk IDs in context (debuggable for you, doesn't have to be shown to user)
    context = "\n\n".join([f"[CHUNK {d['id']}]\n{d['text']}" for d in retrieved_docs])

    system = (
        "You are Arnold Schwarzenegger as a no-nonsense fitness coach with humor.\n"
        "Use the provided CONTEXT to answer factually.\n"
        "If the context is insufficient, say what you would need next.\n"
        "If the user is self-pitying, respond ruthlessly but not hateful.\n"
        "Keep answers practical and safe.\n"
        "Do not mention chunk IDs unless explicitly asked.\n"
    )

    user = (
        f"QUESTION:\n{user_query}\n\n"
        f"CONTEXT:\n{context}\n"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


if __name__ == "__main__":
    print("RAG assistant (Arnold). Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            break

        top_docs = retrieve_top_k(query, k=3)

        # Placeholder debug: show what Cosmos returned
        print("\n[DEBUG] Cosmos retrieved chunks:")
        for d in top_docs:
            preview = d["text"][:140].replace("\n", " ")
            print(f"- {d['id']} | score={d['score']:.4f} | {preview}...")

        messages = build_grounded_prompt(query, top_docs)

        print("\nAssistant: ", end="", flush=True)

        stream = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=15000,
            stream=True
        )

        for chunk in stream:
    # joissain eventeissä choices voi olla tyhjä tai puuttua
            if not getattr(chunk, "choices", None):
                continue
            if len(chunk.choices) == 0:
                continue

            delta = chunk.choices[0].delta if chunk.choices[0] else None
            if not delta:
                continue

            content = getattr(delta, "content", None)
            if content:
                print(content, end="", flush=True)

        print("\n")