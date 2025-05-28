from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_mistralai import ChatMistralAI
import os

# Load environment variables from .env
load_dotenv()

# Retrieve the MISTRAL_API_KEY from environment variables
mistral_api_key = os.getenv("MISTRAL_API_KEY")

# Create a ChatOpenAI model
model = ChatMistralAI(model="mistral-small", api_key=mistral_api_key)

# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Create individual runnables (steps in the chain)
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# Create the RunnableSequence (equivalent to the LCEL chain)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Run the chain
response = chain.invoke({"topic": "lawyers", "joke_count": 3})

# Output
print(response)