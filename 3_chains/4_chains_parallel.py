from dotenv import load_dotenv
import os

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_mistralai import ChatMistralAI

# Load environment variables from .env
load_dotenv()
# Retrieve the MISTRAL_API_KEY from environment variables
mistral_api_key = os.getenv("MISTRAL_API_KEY")

# Create a ChatMistralAI model
model = ChatMistralAI(
    model="mistral-small",
    api_key=mistral_api_key,
)

# Define the main prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer."),
        ("human", "List the main features of the product {product_name}."),
    ]
)

# Define pros analysis step
def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            ("human", "Given these features: {features}, list the pros of these features."),
        ]
    )
    return pros_template.format_prompt(features=features).to_messages()

# Define cons analysis step
def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            ("human", "Given these features: {features}, list the cons of these features."),
        ]
    )
    return cons_template.format_prompt(features=features).to_messages()

# Combine pros and cons into a final review
def combine_pros_cons(branches):
    pros = branches.get("pros", "No pros found")
    cons = branches.get("cons", "No cons found")
    return f"Pros:\n{pros}\n\nCons:\n{cons}"

# Define branch chains
pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x))
    | model
    | StrOutputParser()
)
cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x))
    | model
    | StrOutputParser()
)

# Create the RunnableParallel chain
chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(
        branches={
            "pros": pros_branch_chain,
            "cons": cons_branch_chain,
        }
    )
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]))
)

# Run the chain
try:
    response = chain.invoke({"product_name": "macbook pro 2023"})
    # Output
    print(response)
except Exception as e:
    print(f"Error occurred: {e}")