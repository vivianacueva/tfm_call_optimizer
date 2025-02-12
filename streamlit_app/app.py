# Step 1: Install Required Packages
pip install streamlit openai pinecone-client langchain



# Step 2: Create streamlit app
import streamlit as st
import openai
import pinecone
import os
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Load API keys
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set this in your environment variables
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")  # Replace with Pinecone region

# Connect to Pinecone
index_name = "transcriptions-index"
index = pinecone.Index(index_name)

# Define LangChain retriever
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_store = Pinecone(index, embeddings)
retriever = vector_store.as_retriever()

# LLM setup (DeepSeek or OpenAI GPT-4)
llm = ChatOpenAI(model_name="gpt-4")  # Replace with "deepseek" if using DeepSeek

# RAG Pipeline (Retrieval + LLM)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"  # Retrieves documents and passes them to the model
)

# Streamlit UI
st.title("📞 Call Center Chatbot - RAG-Powered")

# User Input
query = st.text_input("Ask a question about the calls:")

if query:
    response = qa_chain.run(query)
    st.write("### 🤖 AI Response:")
    st.write(response)



# Step 3: Run the Chatbot
streamlit run streamlit_app/app.py
