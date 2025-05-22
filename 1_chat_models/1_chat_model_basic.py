# Chat Model Documents: https://python.langchain.com/docs/integrations/chat/
# MistralAI Chat Model Documents: https://python.langchain.com/docs/integrations/chat/mistralai/


from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the MISTRAL_API_KEY from environment variables
mistral_api_key = os.getenv("MISTRAL_API_KEY")

# Initialize the ChatMistralAI model
llm = ChatMistralAI(
    model="mistral-small",
    api_key=mistral_api_key,
    temperature=0.1,
    max_tokens=256,
)

# Invoke the model with a query
ai_msg = llm.invoke("What is the capital of UK?")
print("Full response:")
print(ai_msg)
print("Content only:")
print(ai_msg.content)