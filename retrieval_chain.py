from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents

docs = get_documents_from_web("https://en.wikipedia.org/wiki/LCEL")

model = ChatOpenAI(
    api_key=api_key,
    model="gpt-3.5-turbo",
    temperature=0.4
)

prompt = ChatPromptTemplate.from_template("""
    Answer the user's question:
    Context: {context}    
    Question: {input}
""")

chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt
)

# chain = prompt | model
print(type(chain))

response = chain.invoke({
    "input": "What is LCEL?",
    "context": [docs]
})

print(response)


