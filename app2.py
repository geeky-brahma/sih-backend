import requests
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Check if GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the image from URL
def load_image(img_url):
    response = requests.get(img_url, stream=True)
    image = Image.open(response.raw).convert("RGB")
    return image

# Generate a conditional caption with context
def generate_conditional_caption(image, prompt, question):
    # Create a prompt that includes context and question
    full_prompt = f"{prompt}. Describe this image and answer the question: {question}"
    inputs = processor(image, full_prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Answer a question based on the image caption
def answer_question(caption, question):
    # Use a more suitable QA model
    qa_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
    answer = qa_model(question=question, context=caption)
    return answer['answer']

# Main function to run image captioning and QA
def main(img_url, question, prompt="a photography of"):
    # Load and process the image
    image = load_image(img_url)
    
    # Generate a conditional caption
    conditional_caption = generate_conditional_caption(image, prompt, question)
    print("Conditional Caption:", conditional_caption)
    
    # Answer the question based on the conditional caption
    answer = answer_question(conditional_caption, question)
    print("Answer to the question:", answer)

# Example usage
if __name__ == "__main__":
    img_url = 'https://www.shutterstock.com/image-photo/young-happy-schoolboy-using-computer-600nw-1075168769.jpg'  # Replace with your image URL
    question = "What is the image about?"  # Replace with your question
    main(img_url, question)
