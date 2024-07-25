## Given input contains 6 images.
## Summarize in 100 words each image with specific clinical issues

# streamlit run medical_assistant1.py --server.maxUploadSize 512
# if submit_button or text_area:

import streamlit as st
import os, textwrap
import json
import google.generativeai as gai
from gtts import gTTS
from tempfile import NamedTemporaryFile
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
from playsound import playsound

# Load environment variables
load_dotenv()
token = os.getenv("GEMINI_API_KEY")
gai.configure(api_key=token)

# Replace 'gemini-pro-vision' with your actual model name if different
#MODEL_NAME = 'gemini-pro-vision'
MODEL_NAME = 'gemini-1.5-flash'

# Set up the model configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_output_tokens": 1024,
    "stop_sequences": ['\n'],
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_LOW_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_LOW_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_LOW_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_LOW_AND_ABOVE"},
]
def create_prompt(user_input: str) -> str:
    base_prompt = {
        "messages": [
            {
                "role": "system",
                "content": "You are medical assistant to physician. \
                Your task is to understand images related to accurate medical conditions, \
                refer accurately doctor group name who specializes \
                in medical conditions or issues helpful to treat the problem.\
                Read medical reports accurate, \
                translate image objects to English text."
            },
            {
                "role": "user",
                "content": "As medical assistant, analyze the attached image \
                and provide accurate insights based on the described tasks."
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
    }
    return json.dumps(base_prompt)

# Function to write chat history to a JSON file
def write_to_json(chat_history):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"medical_assistant_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(chat_history, f)

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with NamedTemporaryFile(delete=False, suffix='.mp3') as f:
        tts.save(f.name)
        return f.name

def clean(text):
    text = text.replace("*", "")
    return text

# Initialize Streamlit app
st.title("Medical Assistant")
st.subheader("Disclaimer::")
st.text("Accurate measurement from image is not possible.")
st.text("Requires further investigation by a qualified physician.")
st.text("Information generated is only to assist and may have error(s).")
st.text("Information should not be a substitute for professional medical advice.")
st.text("Always consult with a doctor for diagnosis and treatment of a medical condition.")

# Model allocation - Model to be loaded only once during app execution
#@st.cache_data
@st.cache_resource
def load_model():
    try:
        model = gai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        return model
    except Exception as e:
        st.error(f"An error occurred while creating the model: {e}")

model = load_model()

# Chat history initialization
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Initialize the input field in session state
if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ''

# Define play_audio_button here
play_audio_button = st.button("Play Audio")

# Define stop_button here
stop_button = st.button("Remove Audio File")

# Image upload section
uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Chat input section

    text_area = st.text_area("Input: ", value=st.session_state['input_text'], key="input")
    submit_button = st.button("Ask your question")

    # Ensure processing only happens when submit button is pressed
    if submit_button or text_area:
        try:
            # Add user query to session chat history
            st.session_state['chat_history'].append(("You", text_area))

            # Generate response
            combined_prompt = create_prompt(text_area)
            response = model.generate_content([combined_prompt, img])

            # Display response
            st.subheader("Reply from Bot:")
            st.write(response.text)
            st.session_state['input_text'] = ''
            # Save response.text as a file
            with open('response.txt', 'w') as myfile:
                data = response.text.replace('\n', '')
                data = str(data)
                tts = gTTS(text=data, lang='en', slow=False)
                tts.save("audio_response_file.mp3")
                myfile.write(data)

            # Add bot response to session chat history
            st.session_state['chat_history'].append(("Reply from Bot::", response.text))

            # Check if chat history length exceeds 10 words per message and write to JSON if necessary
            if len(text_area.split()) > 10 or len(response.text.split()) > 10:
                write_to_json(st.session_state['chat_history'])
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Play audio when play audio button is clicked
if play_audio_button:
    if st.session_state['chat_history']:
        latest_response = st.session_state['chat_history'][-1][1]
        latest_response = clean(latest_response)
        audio_path = text_to_speech(latest_response)
        playsound(audio_path)

# Display chat history
if st.session_state['chat_history']:
    st.subheader("Chat History")
    for sender, message in st.session_state['chat_history']:
        st.text(f"{sender}: {message}")

# Exit button
if stop_button:
    if os.path.exists("audio_response_file.mp3"):
        os.remove("audio_response_file.mp3")
    st.rerun()
    st.stop()
