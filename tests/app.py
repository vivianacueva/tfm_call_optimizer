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

# ✅ Define Streamlit UI
st.title("📞 Call Analyst Assistant")

# Sidebar: Call ID Selection
call_id = st.sidebar.text_input("📌 Call ID:", "call_002")  # Default: call_002

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

    # ✅ Display Assistant Response
    chatbot_response = response["result"]
    st.session_state.messages.append({"role": "assistant", "content": chatbot_response})
    with st.chat_message("assistant"):
        st.markdown(chatbot_response)
