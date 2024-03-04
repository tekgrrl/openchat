from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(documents)

    return splitDocs

def create_vector_store(docs):
    embedding = OllamaEmbeddings()
    vectoreStore = FAISS.from_documents(docs, embedding=embedding)
    return vectoreStore

def create_chain(vectorStore):
    model = Ollama(
        model="mistral:latest",
        temperature=0.4
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Anser the user's question based on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vectorStore.as_retriever(searhc_kwargs={"k": 3})
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.")
    ])

    # Wraps the original retriever with a history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )
    retrieval_chain = create_retrieval_chain(history_aware_retriever, chain) # Was using the unwrapped retriever here

    return retrieval_chain

def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "input": question,
        "chat_history": chat_history
    })

    return response["answer"]

if __name__ == '__main__':
    print("here") # doesn't prompt without this?
    docs = get_documents_from_web("https://docs.kanaries.net/topics/LangChain/langchain-document-loader")
    vectorStore = create_vector_store(docs)
    chain = create_chain(vectorStore)

    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        response = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("assistant:", response)