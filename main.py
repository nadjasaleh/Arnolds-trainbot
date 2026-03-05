from langchain_text_splitters import RecursiveCharacterTextSplitter
from azure.ai.contentunderstanding import ContentUnderstandingClient
from azure.ai.contentunderstanding.models import AnalysisInput
from azure.core.credentials import AzureKeyCredential
from azure.cosmos import CosmosClient, PartitionKey
import os
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
endpoint = os.getenv("CONTENT_UNDERSTANDING_ENDPOINT") 
key = os.getenv("CONTENT_UNDERSTANDING_PRIMARY_KEY")
client = ContentUnderstandingClient(endpoint=endpoint, credential=AzureKeyCredential(key))

cosmos_endpoint = os.getenv("COSMOS_ENDPOINT_arnord")
cosmos_key = os.getenv("COSMOS_KEY_arnord")
cosmos_client = CosmosClient(cosmos_endpoint, cosmos_key)
COSMOS_DB_ID = os.getenv("COSMOS_DB_ID", "AI_Database")
COSMOS_CONTAINER_ID = os.getenv("COSMOS_CONTAINER_ID", "EmbeddingsContainer")
database = cosmos_client.get_database_client(COSMOS_DB_ID)
container = database.get_container_client(COSMOS_CONTAINER_ID)

CHAT_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT")
if not CHAT_MODEL: 
    raise RuntimeError("AZURE_OPENAI_DEPLOYMENT missing (.env)")

EMBED_MODEL = "text-embedding-ada-002"

DOCS = [
    {
        "id": "doc-001",
        "title": "Resetting your password",
        "text": "Use the 'Forgot password' link on the login page. You'll receive an email to set a new password. "
                "Support cannot reset your password manually for security reasons."
    },
    {
        "id": "doc-002",
        "title": "Duplicate charges and billing retries",
        "text": "Duplicate charges may occur if a payment attempt is retried by your bank or payment processor. "
                "Check Billing → Invoices. If you see two invoices for the same period, contact support with invoice IDs."
    },
    {
        "id": "doc-003",
        "title": "Refund policy (general)",
        "text": "Refunds are handled case-by-case. If you believe a charge is incorrect, contact support and include invoice ID, "
                "billing month, and description of the issue. We cannot promise refunds without review."
    },
    {
        "id": "doc-004",
        "title": "Service outages troubleshooting",
        "text": "Check the status page first. If the service is down, incident updates will appear there. "
                "If status is normal but you face issues, try logging out/in and verifying network connectivity."
    },
    {
        "id": "doc-005",
        "title": "Account locked after failed sign-ins",
        "text": "After multiple failed login attempts, accounts may be temporarily locked. Wait 15 minutes and try again or use password reset."
    },
    {
        "id": "doc-006",
        "title": "Managing API keys safely",
        "text": "API keys must be kept secret. Never share them in chat. Rotate keys regularly. If you suspect a leak, revoke the key immediately."
    },
    {
        "id": "doc-007",
        "title": "Two-factor authentication (2FA)",
        "text": "If 2FA is enabled, you'll need an authenticator code. If you lost access to your 2FA device, use account recovery options. "
                "Support will never ask for your OTP codes."
    },
    {
        "id": "doc-008",
        "title": "Exporting customer data",
        "text": "Go to Settings → Data Export. Exports are delivered as a downloadable file. Large exports may take several minutes to generate."
    },
    {
        "id": "doc-009",
        "title": "Canceling subscription",
        "text": "To cancel: Settings → Billing → Subscription → Cancel. Access remains until the end of the billing period."
    },
    {
        "id": "doc-010",
        "title": "Changing plan tiers",
        "text": "Upgrade or downgrade via Settings → Billing → Plan. Upgrades take effect immediately; downgrades apply at next renewal."
    },
    {
        "id": "doc-011",
        "title": "User roles and permissions",
        "text": "Admins can manage users and billing settings. Standard users cannot access billing. Use Roles settings to assign permissions."
    },
]

def get_embedding(text: str) -> list[float]:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def build_doc_matrix(docs):
    vectors = []
    for d in docs:
        combined = f"{d['title']}\n{d['text']}"
        vectors.append(get_embedding(combined))
    return np.array(vectors, dtype=np.float32)

DOC_MATRIX = build_doc_matrix(DOCS)

def retrieve_top_k(query: str, k: int = 3):
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
            "title": "Book chunk",
            "text": item.get("text", "")
        })

    return results

def build_grounded_prompt(user_query: str, retrieved_docs: list[dict]) -> list[dict]:
    sources_block = "\n\n".join(
        [f"[{d['id']}] {d['title']}\n{d['text']}" for d in retrieved_docs]
    )

    system = (
        "You are Arnold Schwarzenegger as a no-nonsense fitness coach with humor.\n"
        "Answer using ONLY the provided SOURCES.\n"
        "If the sources are insufficient, say what's missing and what to do next.\n"
        "If the user is self-pitying, respond ruthlessly but not hateful.\n"
        "End with: Sources: <ids>\n"
    )

    user = (
        f"USER QUESTION:\n{user_query}\n\n"
        f"SOURCES:\n{sources_block}\n\n"
        "INSTRUCTIONS:\n"
        "- Answer in 3–7 sentences.\n"
        "- Do not invent policies or features.\n"
        "- If user asks for credentials, refuse.\n"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

if __name__ == "__main__":
    print("RAG assistant (NimbusCRM). Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            break

        top_docs = retrieve_top_k(query, k=3)

        print("\n[Retrieved top documents]")
        for d in top_docs:
            print(f"- {d['id']} | {d['title']} (score={d['score']:.3f})")

        messages = build_grounded_prompt(query, top_docs)

        resp = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=400,
        )

        answer = resp.choices[0].message.content
        print("\nAssistant:", answer, "\n")