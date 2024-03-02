
import gradio as gr
from ollama import Client

client = Client(host='http://localhost:11434')

conversation_history = []


def chat(prompt):
    conversation_history.append(prompt)
    prompt_with_history = '\n'.join(conversation_history)

    response = client.chat(model='mario:latest', 
                           messages=[
                            {'role': 'user', 
                             'keep_alive': '-1',
    
                             'content': prompt_with_history
                             }])
    answer = response['message']['content']

    conversation_history.append(answer)

    print(conversation_history)

    return answer


iface = gr.Interface(fn=chat, inputs=gr.Textbox(
    lines=2, placeholder="Enter your prompt here..."), outputs="text")
iface.launch()
