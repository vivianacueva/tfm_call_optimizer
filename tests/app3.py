import os
import streamlit as st
from utils import (
    load_env_vars, initialize_llm, initialize_pinecone_client, 
    get_pinecone_index, initialize_embeddings, initialize_vectorstore,
    create_retriever, create_prompt, initialize_rag_chain
)

# ✅ Load environment variables
HUGGINGFACE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_V2 = load_env_vars()

# ✅ Initialize Hugging Face LLM
llm = initialize_llm(HUGGINGFACE_API_KEY)

# ✅ Initialize Pinecone Client and Index
pinecone_client = initialize_pinecone_client(PINECONE_API_KEY)
index = get_pinecone_index(pinecone_client, PINECONE_INDEX_V2)

# ✅ Initialize Embeddings
huggingface_embeddings = initialize_embeddings()

# ✅ Initialize Vector Store
vectorstore = initialize_vectorstore(index, huggingface_embeddings)

# ✅ Streamlit UI Header
st.set_page_config(page_title="Call Analyst Assistant", page_icon="📞", layout="centered")
st.title("📞 Call Analyst Assistant")

# 📌 Sidebar: Call ID Selection
call_id = st.sidebar.text_input("📌 Call ID:", "call_002")

# ✅ Create Retriever based on Selected Call ID
retriever = create_retriever(vectorstore, call_id)

# ✅ Create Prompt
PROMPT = create_prompt()

# ✅ Initialize RAG Chain
retrievalQA = initialize_rag_chain(llm, retriever, PROMPT)

# ✅ Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ✅ User Input for Chatbot
user_query = st.chat_input("Ask a question about the selected call...")

if user_query:
    # ✅ Display User Message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # ✅ Get Chatbot Response
    response = retrievalQA.invoke({"query": user_query})

    # ✅ Extract only the useful part
    raw_output = response["result"].strip()
    if "Useful Answer:" in raw_output:
        chatbot_response = raw_output.split("Useful Answer:")[-1].strip()
    else:
        chatbot_response = raw_output  # fallback if label missing

    if not chatbot_response:
        chatbot_response = "⚠ No useful answer was found for the selected call."

    # ✅ Store and display response cleanly
    st.session_state.messages.append({"role": "assistant", "content": chatbot_response})
    with st.chat_message("assistant"):
        st.markdown(f"""
        <div style='padding: 1rem; background-color: #f5f5f5; border-left: 6px solid #4CAF50;'>
            <b>Answer:</b><br>{chatbot_response}
        </div>
        """, unsafe_allow_html=True)

# ✅ Hide elements to clean the interface
hide_streamlit_style = """
    <style>
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    .stTextInput > div > div > input {font-weight: bold; font-size: 16px;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
