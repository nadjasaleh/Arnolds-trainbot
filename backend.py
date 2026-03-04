from flask import Flask, request, jsonify
from flask_cors import CORS
from azure.cosmos import CosmosClient
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app) # This allows your HTML file to communicate with this Python server

# --- 1. Initialize Clients (Runs once when server starts) ---
openai_client = AzureOpenAI(
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_API_KEY"),
)

cosmos_client = CosmosClient(os.getenv("COSMOS_ENDPOINT_arnord"), os.getenv("COSMOS_KEY_arnord"))
database = cosmos_client.get_database_client("AI_Database")
container = database.get_container_client("EmbeddingsContainer")


# --- 2. The API Endpoint ---
@app.route('/chat', methods=['POST'])
def chat():
    # Get the user's question from the HTML frontend
    data = request.json
    search_query = data.get("query", "")

    if not search_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Embed the user's search query
        query_embedding = openai_client.embeddings.create(
            input=search_query, 
            model="text-embedding-ada-002"
        ).data[0].embedding

        # Retrieve info from database
        query = """
            SELECT TOP 3 c.text, VectorDistance(c.embedding, @query_vector) AS similarity_score
            FROM c
            ORDER BY VectorDistance(c.embedding, @query_vector)
        """
        parameters = [{"name": "@query_vector", "value": query_embedding}]
        results = container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True)

        top_chunks = [item["text"] for item in results]
        context = "\n\n".join(top_chunks)

        # Augment and generate
        prompt = f"""
        You are Arnold Schwarzenegger (the fitness expert).
        Use the context below to answer the question.
        IMPORTANT: If you include a GIF from the context, you MUST format it as a Markdown image using an exclamation mark exactly like this: ![GIF Title](URL)
        Context:
        ---{context}---
        Question:
        ---{search_query}---
        """

        response = openai_client.chat.completions.create(
            model="gpt-4.1", # Ensure this matches your Azure Deployment Name
            messages=[
                {"role": "system", "content": """
                                                    You are Arnold Scwarzenegger, and you are a fitness coach. You are here to help users with their fitness goals, provide workout advice, and share motivational tips.
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

                                                    """},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )

        # Send the answer back to the HTML frontend
        answer = response.choices[0].message.content
        return jsonify({"response": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Starts the server on http://127.0.0.1:5000
    app.run(debug=True, port=5000)