from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage



"""
# #PART 1: Create a ChatPromptTemplate using a template string
prompt_template = "Tell me a joke about {topic}."
# Create a ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template(prompt_template)

# Print the prompt template
prompt = prompt_template.invoke({"topic": "cats"})
print(prompt)
"""


'''
# # PART 2: Prompt with Multiple Placeholders
prompt_template = """
System: You are a {role} assistant.
Human: Help me  {task}
Assistant: I'll help you with that. 
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
prompt = prompt.invoke({"role": "history", "task": "find the capital of UK"})
print(prompt)

print("---------------------")

# # PART 3: Prompt with System and Human Messages (Using Tuples) or #Chat Prompt Templates (For Chat Models) 
# Create multi-role conversation template
template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} assistant."),
    ("human", "Help me with {task}."),
    ("assistant", "I'll help you with that!"),
    ("human", "{follow_up}")
])

# Format and get message objects
messages = template.format_messages(
    role="writing",
    task="creating a blog post",
    follow_up="What's the best structure?"
)
print(messages)
'''

"""
# # PART 4: Prompt with System and Human Messages (Using Tuples) or #Chat Prompt Templates (For Chat Models)

message = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

prompt = ChatPromptTemplate.from_messages(message)
prompt = prompt.invoke({"topic": "lawyer", "joke_count": 3})
print(prompt)

"""

# This does NOT work:
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    HumanMessage(content="Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)