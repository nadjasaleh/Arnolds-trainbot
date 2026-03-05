
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

arnold_gifs = [{'id': 1, 'title': 'Arnold Approval', 'intent': 'approval', 'tags': ['arnold', 'ok', 'approval', 'got it'], 'page_url': 'https://giphy.com/gifs/arnold-xGezQibMMkmTS', 'gif_url': 'https://media.giphy.com/media/xGezQibMMkmTS/giphy.gif'}, 
               {'id': 2, 'title': 'Arnold Terminator Classic', 'intent': 'terminator_classic', 'tags': ['terminator', 'arnold', 'classic'], 'page_url': 'https://giphy.com/gifs/conan-terminator-arnold-12bamrdTlRBOyQ', 'gif_url': 'https://media.giphy.com/media/12bamrdTlRBOyQ/giphy.gif'}, 
               {'id': 3, 'title': 'Arnold Holiday Fun', 'intent': 'holiday_fun', 'tags': ['christmas', 'jingle all the way', 'arnold'], 'page_url': 'https://giphy.com/gifs/filmeditor-christmas-movies-jingle-all-the-way-l2YWia4nRCeE92ZJC', 'gif_url': 'https://media.giphy.com/media/l2YWia4nRCeE92ZJC/giphy.gif'},
               {'id': 4, 'title': 'Arnold Excited', 'intent': 'excitement', 'tags': ['excited', 'hype', 'energy'], 'page_url': 'https://giphy.com/gifs/filmeditor-christmas-movies-jingle-all-the-way-3ofT5xJtfXgV2aqOyI', 'gif_url': 'https://media.giphy.com/media/3ofT5xJtfXgV2aqOyI/giphy.gif'},
               {'id': 5, 'title': 'Arnold Cool Confidence', 'intent': 'cool_confident', 'tags': ['cool', 'confidence', 'arnold'], 'page_url': 'https://giphy.com/gifs/okkultmotionpictures-arnold-schwarzenegger-artists-on-tumblr-T9YdDlG5gHj6U', 'gif_url': 'https://media.giphy.com/media/T9YdDlG5gHj6U/giphy.gif'},
               {'id': 6, 'title': 'Arnold Intense', 'intent': 'intensity', 'tags': ['serious', 'focus', 'intense'], 'page_url': 'https://giphy.com/gifs/schwarzenegger-RPypvDlvWvfy0', 'gif_url': 'https://media.giphy.com/media/RPypvDlvWvfy0/giphy.gif'},
               {'id': 7, 'title': 'Arnold Call To Action', 'intent': 'call_to_action', 'tags': ['do it', 'action', 'reminder'], 'page_url': 'https://giphy.com/gifs/abcnetwork-vomo-vote-or-miss-out-SXHNorUYlo6VirIshz', 'gif_url': 'https://media.giphy.com/media/SXHNorUYlo6VirIshz/giphy.gif'},
               {'id': 8, 'title': 'Arnold Workout Motivation', 'intent': 'workout_motivation', 'tags': ['gym', 'workout', 'weightlifting'], 'page_url': 'https://giphy.com/gifs/fitness-arnold-schwarzenegger-weightlifting-9vYyNGB3kKU1O', 'gif_url': 'https://media.giphy.com/media/9vYyNGB3kKU1O/giphy.gif'},
               {'id': 9, 'title': 'Arnold Flex', 'intent': 'flex_pride', 'tags': ['muscles', 'flex', 'bodybuilding'], 'page_url': 'https://giphy.com/gifs/muscles-arnold-schwarzenegger-g4zFBruwBOHaE', 'gif_url': 'https://media.giphy.com/media/g4zFBruwBOHaE/giphy.gif'}, 
               {'id': 10, 'title': 'Arnold Montage', 'intent': 'montage_hype', 'tags': ['hype', 'montage', 'energy'], 'page_url': 'https://giphy.com/gifs/youtube-schwarzenegger-mix-TIvv7HIMVzbZm', 'gif_url': 'https://media.giphy.com/media/TIvv7HIMVzbZm/giphy.gif'}, 
               {'id': 11, 'title': 'Arnold Surprise', 'intent': 'surprise', 'tags': ['surprised', 'reaction'], 'page_url': 'https://giphy.com/gifs/om-arnold-4y7peyxlXyBd6', 'gif_url': 'https://media.giphy.com/media/4y7peyxlXyBd6/giphy.gif'}, 
               {'id': 12, 'title': 'Arnold Smile', 'intent': 'smile_positive', 'tags': ['smile', 'positive', 'friendly'], 'page_url': 'https://giphy.com/gifs/haydiroket-smile-smiling-ice-cream-12gRrmtMlhC9O', 'gif_url': 'https://media.giphy.com/media/12gRrmtMlhC9O/giphy.gif'}, 
               {'id': 13, 'title': 'Arnold Rage Mode', 'intent': 'rage_mode', 'tags': ['commando', 'rage', 'action'], 'page_url': 'https://giphy.com/gifs/arnold-schwarzenegger-commando-missle-launcher-IsmtGDdJfpWwg', 'gif_url': 'https://media.giphy.com/media/IsmtGDdJfpWwg/giphy.gif'}]

arnold_quotes = [
    {"quote": "I don't just want to be a bodybuilder, I want to be a legend."},
    {"quote": "The last three or four reps is what makes the muscle grow. This area of pain divides a champion from someone who is not a champion."},
    {"quote": "For me, life is continuously being hungry. The meaning of life is not simply to exist, to survive, but to move ahead, to go up, to achieve, to conquer."},
    {"quote": "You have to remember something: Everybody pities the weak; jealousy you have to earn."},
    {"quote": "The resistance that you fight physically in the gym and the resistance that you fight in life can only build a strong character."},
    {"quote": "What we face may look insurmountable. But I learned something from all those years of training and competing. I learned something from all those sets and reps when I didn’t think I could lift another ounce of weight. What I learned is that we are always stronger than we know."},
    {"quote": "The worst thing I can be is the same as everybody else. I hate that."},
    {"quote": "Strength does not come from winning. Your struggles develop your strengths."},
    {"quote": "Failure is not an option. Everyone has to succeed."},
    {"quote": "You can’t climb the ladder of success with your hands in your pockets."},
    {"quote": "If you don’t find the time, if you don’t do the work, you don’t get the results."},
    {"quote": "Trust yourself. Break some rules. Don’t be afraid to fail."},
    {"quote": "The mind is the limit. As long as the mind can envision the fact that you can do something, you can do it."},
    {"quote": "What is the point of being on this Earth if you are going to be like everyone else?"},
    {"quote": "the resistance that you fight physically in the gym and the resistance that you fight in life can only build a strong character."},
    {"quote": "For me, life is continuously being hungry."},
    {"quote": "The last three or four reps is what makes the muscle grow."},
    {"quote": "You have to remember something: Everybody pities the weak; jealousy you have to earn."},
    {"quote": "No pain, no gain."},
    {"quote": "Just remember, you can’t climb the ladder of success with your hands in your pockets."},
    {"quote": "When you go through hardships and decide not to surrender, that is strength."},
    {"quote": "The worst thing I can be is the same as everybody else."},
    {"quote": "If you want to turn a vision into reality, you have to give 100% and never stop believing in your dream."},
    {"quote": "I didn’t mind basic training. It taught me that something that seems impossible at the start can be achieved."},
    {"quote": "To be successful, however, you must be brutal with yourself and focus on the flaws."},
    {"quote": "You can scream at me, call me for dinner, but I’m not coming home until I finish my workout."}
]

documents = [
    {"name": "LeavePolicy.txt", "content": "Employees are entitled to 20 days of paid leave annually. Leave requests should be submitted at least two weeks in advance through the HR portal. Unused leave may be carried over to the next year up to a maximum of 5 days. Special leave, such as bereavement or parental leave, is available upon request and subject to approval."},
    {"name": "WorkFromHomePolicy.txt", "content": "Employees may work from home up to 3 days a week, provided their duties allow for remote work. Approval from the line manager is required. Employees must ensure a productive work environment and be available during core business hours. All company data accessed remotely must be handled securely."},
    {"name": "CodeOfConduct.txt", "content": "All employees must adhere to respectful communication, maintain professionalism, and foster an inclusive workplace. Discrimination, harassment, or any form of misconduct will not be tolerated. Employees are expected to report any violations to HR or their supervisor promptly."},
    {"name": "ITPolicy.txt", "content": "Company devices must be secured and used for work only. Employees are responsible for updating software regularly and reporting any security incidents immediately. Personal use of company devices should be minimal and must not interfere with work responsibilities."},
    {"name": "Benefits.txt", "content": "Health insurance, dental coverage, and gym reimbursement are provided to all full-time employees. Additional benefits include mental health support, paid parental leave, and access to wellness programs. Details of each benefit can be found in the employee handbook or by contacting HR."},
    {"name": "ProbationPeriod.txt", "content": "New hires undergo a 3-month probation period during which their performance and cultural fit are evaluated. Feedback will be provided regularly, and successful completion of probation will be confirmed in writing. Extensions to the probation period may be considered if necessary."},
    {"name": "TravelPolicy.txt", "content": "Business travel must be pre-approved by the line manager. Employees should book travel through the company's preferred vendors and adhere to the travel expense guidelines. All receipts must be submitted within 7 days of return for reimbursement."},
    {"name": "PerformanceReview.txt", "content": "Performance is reviewed bi-annually based on KPIs and individual goals. Employees are encouraged to participate actively in the review process and set development objectives. Constructive feedback will be provided, and training opportunities may be recommended."},
    {"name": "ExitPolicy.txt", "content": "Resignation requires a 30-day notice period, submitted in writing to the supervisor and HR. An exit interview will be conducted to gather feedback. All company property must be returned before the final day of employment, and outstanding expenses must be settled."},
    {"name": "OnboardingGuide.txt", "content": "New employees must complete orientation within the first week, which includes training on company policies, IT systems, and team introductions. A mentor will be assigned to assist with the transition. All required documentation must be submitted to HR during this period."},
    {"name": "DressCode.txt", "content": "Employees are expected to dress in business casual attire during office hours. Exceptions may be made for fieldwork or special events. Casual Fridays allow for more relaxed dress, but clothing should remain professional and appropriate for the workplace."},
    {"name": "RemoteWorkGuidelines.txt", "content": "Remote work requires stable internet and availability during core hours. Employees must attend virtual meetings and respond to communications promptly. The company may provide equipment as needed, and all remote work arrangements must be reviewed annually."},
    {"name": "ExpensePolicy.txt", "content": "All expenses must be submitted with receipts within 30 days of incurring the cost. Only business-related expenses are eligible for reimbursement. The finance team reviews all claims, and any discrepancies will be communicated to the employee for clarification."},
    {"name": "DataPrivacy.txt", "content": "Personal data must be handled in accordance with GDPR regulations. Employees are responsible for safeguarding confidential information and reporting any data breaches immediately. Regular training on data privacy is mandatory for all staff."},
    {"name": "MeetingEtiquette.txt", "content": "Meetings should start on time and have a clear agenda circulated in advance. Participants are expected to contribute constructively and respect others’ viewpoints. Action items and minutes should be documented and shared after each meeting."},
    {"name": "SecurityPolicy.txt", "content": "Passwords must be updated every 90 days and not shared with anyone. Employees should lock their computers when away from their desks and report suspicious activity to IT. Regular security awareness training is provided to all staff."},
    {"name": "EqualOpportunity.txt", "content": "The company is committed to providing equal opportunities for all employees regardless of race, gender, age, religion, or disability. Recruitment, promotion, and training decisions are based on merit and business needs. Any concerns should be reported to HR."},
    {"name": "HarassmentPolicy.txt", "content": "Harassment of any kind will not be tolerated and should be reported immediately to HR or a supervisor. All complaints will be investigated promptly and confidentially. Support resources are available for affected employees."},
    {"name": "TrainingAndDevelopment.txt", "content": "Employees are encouraged to attend at least one training session per quarter to enhance their skills. The company offers a variety of internal and external training programs. Requests for additional training can be submitted to the HR department."},
    {"name": "InternetUsage.txt", "content": "Internet access is provided for work purposes; personal browsing should be limited and must not interfere with job responsibilities. Accessing inappropriate or illegal content is strictly prohibited and may result in disciplinary action."},
]

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

#for q in arnold_quotes:
#    embedding = openai_client.embeddings.create(input=q["quote"],model="text-embedding-ada-002").data[0].embedding
#    container.upsert_item({"id": str(uuid.uuid4()), "text": q["quote"],"embedding": embedding}) 

#for doc in documents:
#    embedding = openai_client.embeddings.create(input=f"{doc['name']}\n{doc['content']}",model="text-embedding-ada-002").data[0].embedding
#    container.upsert_item({"id": str(uuid.uuid4()), "text": f"{doc['name']}\n{doc['content']}","embedding": embedding}) 

#for gif in arnold_gifs:
#    embedding = openai_client.embeddings.create(input=f"{gif['title']} {gif['intent']} {' '.join(gif['tags'])}",model="text-embedding-ada-002").data[0].embedding
#    container.upsert_item({"id": str(uuid.uuid4()), "text": f"GIF: {gif['title']} | Intent: {gif['intent']} | URL: {gif['gif_url']}","embedding": embedding}) 

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