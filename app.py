import gradio as gr
import openai
import torch
from PIL import Image

# Use a pipeline as a high-level helper
from transformers import pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
# Replace with your actual OpenAI API key
openai.api_key = "sk-hOLXpm0JVDTDyzRSyO6-eeTw_z9dTDl_NgxbX3kcefT3BlbkFJW3SLYTR9qpSn5nlrSVWohJV5Irk0UnIA45hpVLcrkA"
caption_image = pipeline("image-to-text",
                model="Salesforce/blip-image-captioning-large", device=device)
def caption_my_image(pil_image):
    semantics = caption_image(images=pil_image)[0]['generated_text']
    print(semantics)
    return semantics
def chat_with_llm(message, history):
    # Initialize message history if empty
    if history is None:
        history = []
    pil_image = Image.open("gargoyle-8791108_640.jpg")
    desc=caption_my_image(pil_image)
    print("desc:"+desc)
    # Convert history into format suitable for OpenAI Chat API
    # messages = [{"role": "system", "content": "You are a helpful assistant."}]
    # messages = [{"role": "system", "content": "You are a helpful assistant. You are given a description of a image uploaded by user. You need to answer questions based on it. Description:",}]
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You are given a description of an image uploaded by user. You need to answer questions based on it. Description: " + desc
        }
    ]
    # Add the user and bot messages from history to the messages list
    for user_input, response in history:
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": response})
    
    # Add the latest user message
    messages.append({"role": "user", "content": message})
    
    # Get response from OpenAI's Chat API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150
    )

    
    # Extract the AI's reply
    ai_message = response.choices[0].message['content']
    
    # Append the user message and the AI response to the history
    history.append((message, ai_message))
    
    # Return the updated history to Gradio
    # return history
    return ai_message

# Create a Gradio chat interface
demo = gr.ChatInterface(
    fn=chat_with_llm,
    examples=["hello", "hola", "merhaba"],
    title="AI Chat Bot"
)

# Launch the interface
demo.launch()
