<<<<<<< HEAD
import streamlit as st
from rag_1 import get_rag_response  # Import the RAG system function

# Set the page configuration
st.set_page_config(page_title="RAG System Chat", page_icon=":robot_face:", layout="wide")

# Set the background color and text color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #2e2e2e;
        color: #f0f0f0;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100vh;
    }
    .chat-container {
        flex: 1;
        overflow-y: auto;
        padding: 10px;
    }
    .input-container {
        display: flex;
        align-items: center;
        padding: 10px;
        background-color: #3e3e3e;
        border-radius: 10px;
    }
    .input-container input {
        flex: 1;
        padding: 10px;
        border-radius: 10px;
        border: none;
        margin-right: 10px;
        background-color: #4e4e4e;
        color: #f0f0f0;
    }
    .input-container button {
        padding: 10px;
        border-radius: 10px;
        border: none;
        background-color: #4e4e4e;
        color: #f0f0f0;
    }
    .user-message {
        background-color: #4e4e4e;
        color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        text-align: right;
        align-self: flex-end;
        max-width: 70%;
    }
    .bot-message {
        background-color: #3e3e3e;
        color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        text-align: left;
        align-self: flex-start;
        max-width: 70%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the app
st.title("RAG System Chat")

# Initialize session state for storing messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to handle user input
def handle_input():
    user_input = st.session_state.user_input
    if user_input:
        # Append user message to session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Get response from the RAG system
        response = get_rag_response(user_input)
        st.session_state.messages.append({"role": "bot", "content": response})
        # Clear the input box
        st.session_state.user_input = ""

# Display chat messages
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-message'>{message['content']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Input box for user questions
st.markdown(
    """
    <div class="input-container">
        <input type="text" id="user_input" placeholder="Ask a question..." onkeypress="if(event.key === 'Enter') { document.getElementById('submit_button').click(); }">
        <button id="submit_button" onclick="handle_input()">Submit</button>
    </div>
    """,
    unsafe_allow_html=True
)

# Button to submit the question
if st.button("Submit", key="submit_button"):
    handle_input()
=======
import streamlit as st

# Set the page configuration
st.set_page_config(page_title="RAG System Chat", page_icon=":robot_face:", layout="wide")

# Set the background color and text color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #2e2e2e;
        color: #f0f0f0;
    }
    .stTextInput > div > div > input {
        background-color: #3e3e3e;
        color: #f0f0f0;
    }
    .stButton > button {
        background-color: #4e4e4e;
        color: #f0f0f0;
    }
    .stMarkdown {
        color: #f0f0f0;
    }
    .user-message {
        background-color: #4e4e4e;
        color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        text-align: right;
    }
    .bot-message {
        background-color: #3e3e3e;
        color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        text-align: left;
    }
    .input-container {
        display: flex;
        align-items: center;
    }
    .input-container input {
        flex: 1;
        padding: 10px;
        border-radius: 10px;
        border: none;
        margin-right: 10px;
    }
    .input-container button {
        padding: 10px;
        border-radius: 10px;
        border: none;
        background-color: #4e4e4e;
        color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the app
st.title("RAG System Chat")

# Initialize session state for storing messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to handle user input
def handle_input():
    user_input = st.session_state.user_input
    if user_input:
        # Append user message to session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Here you would call your RAG system to get the response
        response = "This is a placeholder response from the RAG system."
        st.session_state.messages.append({"role": "bot", "content": response})
        # Clear the input box
        st.session_state.user_input = ""

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-message'>{message['content']}</div>", unsafe_allow_html=True)

# Input box for user questions
st.markdown(
    """
    <div class="input-container">
        <input type="text" id="user_input" placeholder="Ask a question..." onkeypress="if(event.key === 'Enter') { document.getElementById('submit_button').click(); }">
        <button id="submit_button" onclick="handle_input()">Submit</button>
    </div>
    """,
    unsafe_allow_html=True
)

# Button to submit the question
if st.button("Submit", key="submit_button"):
    handle_input()
>>>>>>> f716a4b37cc7ee1598b2aaf37726d6dfdb93cf8a
