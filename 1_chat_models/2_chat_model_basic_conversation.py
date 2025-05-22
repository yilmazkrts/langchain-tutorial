from langchain_mistralai import ChatMistralAI

# Load environment variables from .env file
from dotenv import load_dotenv
import os

load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")

# Initialize the ChatMistralAI model
llm = ChatMistralAI(
    model="mistral-small",
    api_key=mistral_api_key,
    temperature=0.1,
)

# Define the messages for the conversation
messages = [
    (
        "system", "You are a helpful history assistant."
    ),
    (
        "human", "What role did London play during the Industrial Revolution?"
    ),
]

# Invoke the model with a query
ai_msg = llm. invoke (messages)
print(f"Answer from AI: {ai_msg.content}")

