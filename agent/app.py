import time
import streamlit as st
from graph import ChatbotGraph

st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="centered")
st.title("💬 DoorDash Support Chatbot")

# Initialize chatbot and conversation history
if "chatbot" not in st.session_state:
    st.session_state.chatbot = ChatbotGraph(thread_id="chat_thread_1")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}

# Replay previous chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box for new messages
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Save user's message first
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response streaming
    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed_response = ""

        # Stream tokens directly from LangGraph
        with st.spinner("🤖 Thinking..."):
            for token in st.session_state.chatbot.ask(user_input):
                streamed_response += token
                placeholder.markdown(streamed_response + "▌")

        # After streaming completes, remove cursor
        placeholder.markdown(streamed_response)
    # Save assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": streamed_response})

# # Assistant response streaming (only if user_input is not None)
# if user_input:
#     with st.chat_message("assistant"):
#         placeholder = st.empty()
#         streamed_response = ""

#         # Show spinner while generating response
#         with st.spinner("🤖 Thinking..."):
#             for token in st.session_state.chatbot.ask(user_input):
#                 streamed_response += token
#                 placeholder.markdown(streamed_response + "▌")

#         # After streaming completes, remove cursor
#         placeholder.markdown(streamed_response)

#     # Save assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": streamed_response})