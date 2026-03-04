from langchain_text_splitters import RecursiveCharacterTextSplitter
from azure.ai.contentunderstanding import ContentUnderstandingClient
from azure.ai.contentunderstanding.models import AnalysisInput
from azure.core.credentials import AzureKeyCredential
from azure.cosmos import CosmosClient, PartitionKey
from openai import AzureOpenAI
from dotenv import load_dotenv
import uuid
import os

load_dotenv()
# read the variables
endpoint = os.getenv("CONTENT_UNDERSTANDING_ENDPOINT") 
key = os.getenv("CONTENT_UNDERSTANDING_PRIMARY_KEY")
client = ContentUnderstandingClient(endpoint=endpoint, credential=AzureKeyCredential(key))

openai_client = AzureOpenAI(
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_API_KEY"),
)

cosmos_endpoint = os.getenv("COSMOS_ENDPOINT_arnord")
cosmos_key = os.getenv("COSMOS_KEY_arnord")
cosmos_client = CosmosClient(cosmos_endpoint, cosmos_key)

# convert from scanned to markdown
file_url = "https://nnty.fun/downloads/books/TAEL/SELF%20IMPROVEMENT/FITNESS/BODYBUILDING/The%20New%20Encyclopedia%20of%20Modern%20Bodybuilding%20-%20Simon%20%26%20Schuster.%20Arnold%20Schwarzenegger%2C%20Bill%20Dobbins%20%281998%29.pdf"
poller = client.begin_analyze(analyzer_id="prebuilt-document", inputs=[AnalysisInput(url=file_url)]) #prebuilt-document  / prebuilt-layout (simpler)
markdown_content = poller.result().contents[0].markdown


# chunk
text_splitter = RecursiveCharacterTextSplitter(
   chunk_size=2000, 
   chunk_overlap=500,
   length_function=len,
   is_separator_regex=False
)

chunks = text_splitter.create_documents([markdown_content])
print("Number of chunks: ", len(chunks))

database = cosmos_client.create_database_if_not_exists(id="AI_Database")
container = database.create_container_if_not_exists(id="EmbeddingsContainer", partition_key=PartitionKey(path="/id"))
print("connected to CosmosDB!")


# embedd and store in cosmosDB # comment out this code because we already have the embeddings in cosmos
#for chunk in chunks:
#    embedding = openai_client.embeddings.create(input=chunk.page_content,model="text-embedding-ada-002").data[0].embedding
#    container.upsert_item({"id": str(uuid.uuid4()), "text": chunk.page_content,"embedding": embedding}) 

print("all embeddings are inserted into CosmosDB!")

# retrive info from database (book)
search_query = "how can i train my shoulders?"
query_embedding = openai_client.embeddings.create(input=search_query,model="text-embedding-ada-002").data[0].embedding

query = """
    SELECT TOP 3 c.text, VectorDistance(c.embedding, @query_vector) AS similarity_score
    FROM c
    ORDER BY VectorDistance(c.embedding, @query_vector)
"""

parameters = [{"name": "@query_vector", "value": query_embedding}]

results = container.query_items(query=query, parameters=parameters,enable_cross_partition_query=True)


top_chunks = [item["text"] for item in results]
context = "\n\n".join(top_chunks)


# augment
prompt = f"""
You are Arnold Schwarzenegger (the fitness expert).
Use the context below to answer the question.
Context:
---{context}---
Question:
---{search_query}---
"""

response = openai_client.chat.completions.create(
    model="gpt-4.1", 
    messages=[
        {"role": "system", "content": "You are Arnold Schwarzenegger (the fitness expert)."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.5
)

print(response.choices[0].message.content)