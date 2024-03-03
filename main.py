
import gradio as gr
from ollama import Client
from langchain.schema import AIMessage, HumanMessage

client = Client(host='http://localhost:11434')

conversation_history = []


def chat(prompt, history):

    history_langchain_format = []

    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))

    history_langchain_format.append(HumanMessage(content=prompt))

    response = client.chat(model='mistral:latest', 
                           messages=[
                            {'role': 'user', 
                             'keep_alive': '-1',
    
                             'content': history_langchain_format
                             }])
    answer = response['message']['content']

    # conversation_history.append(answer)
    # print(conversation_history)

    return answer

iface = gr.ChatInterface(chat).launch()
