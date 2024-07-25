import streamlit as st
import os, json
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
MODEL_NAME = 'gemini-1.5-flash'

# Set up the model configuration
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 16384,
    "stop_sequences": ['time'],
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
                Your task is to understand images related to medical conditions accurately, \
                refer doctor group name who specializes \
                in medical conditions or issues helpful to treat the problem. \
                Read medical reports, translate image text or objects to English text."
            },
            {
                "role": "user",
                "content": "As medical assistant, please analyze the attached image \
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

# Function to remove non-ASCII characters
def remove_non_ascii(s):
    return ''.join(filter(lambda x: ' ' <= x <= '~', s))

# Initialize Streamlit app
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
            border: 5px solid orange;
            padding: 20px;
            border-radius: 10px;
            border: 5px solid orange;
            border-inner: 2px solid pink;
            padding: 20px;
        }
        .main-title {
            color: #f3399CC;
            font-family: 'Arial', sans-serif;
            text-align: center;
            background-color: #ff4500;
            padding: 10px;
            border-radius: 5px;
        }
        .sub-title {
            font-size: 18px;
            color: #ffffff;
            font-family: 'Arial', sans-serif;
            margin-left: 20px;
        }
        .disclaimer-text {
            color: #ffffff;
            font-family: 'Arial', sans-serif;
            margin-top: -5px;
            margin-left: 20px;
            font-size: 14px;
        }
        .chat-history {
            border: 1px solid #dcdcdc;
            padding: 10px;
            border-radius: 5px;
            background-color: #2c3e50;
            color: #ecf0f1;
        }
        .custom-button {
            background-color: #4CAF50; 
            border: none;
            color: white;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            transition-duration: 0.4s;
            cursor: pointer;
        }
        .custom-button:hover {
            background-color: white;
            color: black;
            border: 2px solid #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("<h1 class='main-title'>Medical Assistant</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='sub-title'>Disclaimer:</h2>", unsafe_allow_html=True)
st.markdown("<p class='disclaimer-text'>Accurate measurement from image is not possible. Requires investigation by qualified physician</p>", unsafe_allow_html=True)
st.markdown("<p class='disclaimer-text'>Information generated may have error(s), should not be a substitute for professional medical advice.</p>", unsafe_allow_html=True)
st.markdown("<p class='disclaimer-text'>Always consult with a doctor for diagnosis and treatment of a medical condition.</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Model allocation - Model to be loaded only once during app execution
@st.cache_data
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

# Image upload section
uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Chat input section
    text_area = st.text_area("Input: ", key="input")
    submit_button = st.button("Ask your question", key="ask_question", help="Submit your question")

    if submit_button or text_area:
        try:
            # Add user query to session chat history
            st.session_state['chat_history'].append(("You", text_area))

            # Generate response
            combined_prompt = create_prompt(text_area)
            response = model.generate_content([combined_prompt, img])

            # Filter response.text to only contain ASCII characters between ' ' (space) and '~' (tilde)
            filtered_response_text = response.text

            # Display response
            st.subheader("Clinical assistant response:")
            st.write(filtered_response_text)

            # Save response.text as a file
            with open('response.txt', 'w') as myfile:
                data = filtered_response_text.replace('\n', '')
                tts = gTTS(text=data, lang='en', slow=False)
                tts.save("audio_response_file.mp3")
                myfile.write(data)

            # Add bot response to session chat history
            st.session_state['chat_history'].append(("Reply from Bot:", filtered_response_text))

            # Check if chat history length exceeds 10 words per message and write to JSON if necessary
            if len(text_area.split()) > 10 or len(filtered_response_text.split()) > 10:
                write_to_json(st.session_state['chat_history'])
        except Exception as e:
            st.error(f"An error occurred: {e}")

if st.session_state['chat_history']:
    st.subheader("Chat History")
    # Use a set to track displayed messages and avoid duplicates
    displayed_messages = set()
    for sender, message in st.session_state['chat_history']:
        message_id = f"{sender}: {message}"
        if message_id not in displayed_messages:
            st.markdown(f"<div class='chat-history'><strong>{sender}:</strong> {message}</div>", unsafe_allow_html=True)
            displayed_messages.add(message_id)

# Create a ribbon for audio control buttons
st.markdown("<div class='ribbon'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    play_audio_button = st.button("Play Audio", key="play_audio", help="Play the audio response")
# with col2:
#     pause_audio_button = st.button("Pause Audio", key="pause_audio", help="Pause the audio response")
with col3:
    stop_button = st.button("Remove Audio File", key="remove_audio", help="Remove the audio file")
st.markdown("</div>", unsafe_allow_html=True)

if play_audio_button:
    if st.session_state['chat_history']:
        latest_response = st.session_state['chat_history'][-1][1]
        latest_response = clean(latest_response)
        audio_path = text_to_speech(latest_response)
        playsound(audio_path)

# Exit button
if stop_button:
    if os.path.exists("audio_response_file.mp3"):
        os.remove("audio_response_file.mp3")
    st.rerun()
    st.stop()

