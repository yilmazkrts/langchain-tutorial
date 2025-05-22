from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables from .env file
from dotenv import load_dotenv
import os

load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")

chat_history = []

# Initialize the ChatMistralAI model
llm = ChatMistralAI(
    model="mistral-small",
    api_key=mistral_api_key,
    temperature=0.1,
)

system_message = SystemMessage(
    content="You are a helpful history assistant."
)

chat_history.append(system_message)
print(f"Chat history after adding system message: {chat_history}")

while True:
    query = input("Enter your question (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break

    # Add the human message to the chat history
    human_message = HumanMessage(query)
    chat_history.append(human_message)

    # Invoke the model with the chat history
    ai_msg = llm.invoke(chat_history)
    print(f"Answer from AI: {ai_msg.content}")
    chat_history.append(ai_msg)

    # Print the chat history for debugging
    #print(f"Chat history after adding human message: {chat_history}")

