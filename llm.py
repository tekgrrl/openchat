from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-3.5-turbo"
)

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Generate a list of 10 synonyms for the following word: Return the results as a comma-separated list."),
        ("human", "{input}")
    ]
)

# Create LLM chain
chain = prompt | llm

response = chain.invoke({"input": "happy"})
print(response)