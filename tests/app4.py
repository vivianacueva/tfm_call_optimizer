import streamlit as st
import os
from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Initialize API Keys and Configs
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_V2")

# Initialize LLM
llm = HuggingFaceHub(
    repo_id="deepseek-ai/DeepSeek-R1",
    model_kwargs={"temperature": 0.5, "max_length": 2048},
    huggingfacehub_api_token=HUGGINGFACE_API_KEY
)

# Initialize Pinecone
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone_client.Index(PINECONE_INDEX)

# Embeddings
embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Vectorstore
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# Prompt Template
prompt_template = """
You are an expert auditor analyzing call center transcription calls from custumer support calls.
You need to give useful insights from the questions an audit expert would make so he can understand how the calls went.

Instructions:
1. Answer based on the provided context (delimited by <ctx> </ctx>) and the chat history (delimited by <hs> </hs>) below.
2. If the information is not in the context, respond: "I don't have this information."
3. Provide a concise and precise answer.

Provided Information
-------
<ctx>
Context: {context}
</ctx>
-------
<hs>
Chat History: {chat_history}
</hs>
-------
Question: {question}

Useful Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "chat_history", "question"]
)

# Streamlit UI
st.set_page_config(page_title="Call Analyst Assistant", layout="wide")
st.title("Call Analyst Assistant")

# Sidebar for call ID and reset
st.sidebar.subheader("Call ID:")
call_id = st.sidebar.text_input("", value="call_002")

# Context scope selector
scope = st.sidebar.radio("Context scope:", ["Call only", "PDF only", "Both"])

# Determine metadata filter based on user selection
if scope == "Call only":
    metadata_filter = {"call_id": call_id}
elif scope == "PDF only":
    metadata_filter = {"source": "pdf"}
else:
    metadata_filter = {"$or": [{"call_id": call_id}, {"source": "pdf"}]}

# Memory buffer
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")

# Reset button
if st.sidebar.button("🔄 Reset chat"):
    st.session_state.chat_history = []
    memory.clear()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Create retriever with dynamic filtering
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
        "filter": metadata_filter
    }
)

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt, "memory": memory}
)

# Chat UI
st.divider()

for i, (user, bot) in enumerate(st.session_state.chat_history):
    st.chat_message("user", avatar="✉").markdown(user)
    st.chat_message("assistant", avatar="🤖").markdown(bot)

user_input = st.chat_input("Ask a question about this call or general resolution procedures...")

if user_input:
    response = qa_chain.invoke({"query": user_input})
    answer = response["result"]

    # Update history
    st.session_state.chat_history.append((user_input, answer))

    # Display current turn
    st.chat_message("user", avatar="✉").markdown(user_input)
    st.chat_message("assistant", avatar="🤖").markdown(answer)
