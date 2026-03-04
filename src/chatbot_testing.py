#testing1
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-03-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

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

MODEL = "gpt-4.1"
MAX_TOKENS = 1500
TEMPERATURE = 1.0

# --- Validation lists ---
#BLOCKED_INPUT_KEYWORDS = ["competitor", "lawsuit", "medical advice", "legal advice"]
#BLOCKED_OUTPUT_KEYWORDS = ["my instructions are", "my guardrails", "i was told to"]
print("Chat started! Type 'quit' to exit.\n")

previous_response_id = None

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    # --- Input validation ---
    #if any(word in user_input.lower() for word in BLOCKED_INPUT_KEYWORDS):
     #   print("Assistant: I'm sorry, I can't help with that topic. Is there something else I can help you with?\n")
      #  continue  # skip API call entirely

    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=user_input,
        previous_response_id=previous_response_id,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_TOKENS
    )

    # --- Output validation ---
    #if any(word in response.output_text.lower() for word in BLOCKED_OUTPUT_KEYWORDS):
    #    print("Assistant: I'm sorry, I can't provide that information.\n")
        # don't update previous_response_id so the bad response is not part of context
     #   continue

    print(f"Assistant: {response.output_text}\n")

    # Only update if everything passed validation
    previous_response_id = response.id