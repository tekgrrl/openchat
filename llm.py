from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-3.5-turbo"
)

response = llm.stream("Why is the sky blue?")

for chunk in response: 
    print(chunk.content, end="", flush=True)