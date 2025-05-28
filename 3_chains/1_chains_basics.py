from dotenv import load_dotenv
import os

from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Retrieve the MISTRAL_API_KEY from environment variables
mistral_api_key = os.getenv("MISTRAL_API_KEY")

# Initialize the ChatMistralAI model

llm = ChatMistralAI(
    model="mistral-small",
    api_key=mistral_api_key,
)

# Create a chat prompt template

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)


# Create the combined chain using LangChain Expression Language (LCEL)
chain = chat_prompt | llm | StrOutputParser()
#chain = chat_prompt | llm

# Run the chain
result = chain.invoke({"topic": "lawyers", "joke_count": 3})

# Output
print(result)



