from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
# Load environment variables from .env file
from dotenv import load_dotenv
import os

load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")

llm = ChatMistralAI(
    model="mistral-small",
    api_key=mistral_api_key,
    temperature=0.1,
)

# PART 1: Create a ChatPromptTemplate using a template string
template = "Tell me a joke about {topic}"
prompt_template = ChatPromptTemplate.from_template(template)

# Format the prompt with the topic
formatted_prompt = prompt_template.format(topic="lawyer")

# Invoke the LLM with the formatted prompt
result = llm.invoke(formatted_prompt)
print(result.content)



# PART 2: Prompt with Multiple Placeholders
print("\n----- Prompt with Multiple Placeholders -----\n")
template_multiple = """You are a helpful assistant.
Human: Tell me a {adjective} short story about a {animal}.
Assistant:"""
prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})

result = llm.invoke(prompt)
print(result.content)


# PART 3: Prompt with System and Human Messages (Using Tuples)
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
result = llm.invoke(prompt)
print(result.content)

