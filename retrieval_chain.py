from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(documents)

    return splitDocs

def create_vector_store(docs):
    embedding = OpenAIEmbeddings()
    vectoreStore = FAISS.from_documents(docs, embedding=embedding)
    return vectoreStore

def create_chain(vectorStore):
    model = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
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

    retriever = vectorStore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, chain)

    return retrieval_chain

docs = get_documents_from_web("https://python.langchain.com/docs/expression_language/")
vectorStore = create_vector_store(docs)

chain = create_chain(vectorStore)

response = chain.invoke({
    "input": "What is LCEL?",
})

print(response["answer"])

