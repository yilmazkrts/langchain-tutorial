from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser

from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()
# Retrieve the MISTRAL_API_KEY from environment variables
mistral_api_key = os.getenv("MISTRAL_API_KEY")

# Create a ChatMistralAI model
model = ChatMistralAI(model="mistral-small", api_key=mistral_api_key)

# Define prompt templates

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Create individual runnables (steps in the chain)
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

# Run the chain
response = chain.invoke({"topic": "lawyers", "joke_count": 3})

# Output
print(response)