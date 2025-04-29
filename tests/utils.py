import os
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# ✅ Load environment variables
def load_env_vars():
    load_dotenv()
    huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index = os.getenv("PINECONE_INDEX_V2")
    
    if not huggingface_api_key:
        raise ValueError("❌ Missing Hugging Face API Key. Set it as HUGGINGFACEHUB_API_TOKEN")
    
    if not pinecone_api_key:
        raise ValueError("❌ Missing Pinecone API Key. Set it as PINECONE_API_KEY")

    if not pinecone_index:
        raise ValueError("❌ PINECONE_INDEX_V2 is not set. Check your .env file.")

    return huggingface_api_key, pinecone_api_key, pinecone_index


# ✅ Initialize Hugging Face LLM
def initialize_llm(api_key):
    return HuggingFaceHub(
        repo_id="deepseek-ai/DeepSeek-R1",
        model_kwargs={"temperature": 0.5, "max_length": 2048},
        huggingfacehub_api_token=api_key
    )


# ✅ Initialize Pinecone Client
def initialize_pinecone_client(api_key):
    return Pinecone(api_key=api_key)


# ✅ Retrieve or Create Pinecone Index
def get_pinecone_index(client, index_name):
    existing_indexes = [idx["name"] for idx in client.list_indexes()]
    if index_name not in existing_indexes:
        raise ValueError(f"❌ Index '{index_name}' does not exist in Pinecone. Please create it first.")
    
    return client.Index(index_name)


# ✅ Initialize Hugging Face Embeddings
def initialize_embeddings():
    return HuggingFaceBgeEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


# ✅ Initialize Pinecone Vector Store
def initialize_vectorstore(index, embeddings):
    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )


# ✅ Create Retriever with Metadata Filtering (for a specific call)
def create_retriever(vectorstore, call_id):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,
            "filter": {"call_id": call_id}  # ✅ Filter by call ID
        }
    )


# ✅ Define Prompt Template
def create_prompt():
    prompt_template = """
    You are an expert auditor analyzing call center transcription calls from customer support calls.
    You need to give useful insights from the questions an audit expert would make so he can understand how the calls went.

    Instructions:
    1. Answer based on the provided context (delimited by <ctx> </ctx>) and the chat history (delimited by <hs> </hs>) below.
    2. If the information is not in the context, respond: "I don't have this information."
    3. **Provide a concise and precise answer.**

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
    return PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])


# ✅ Initialize RAG Chain
def initialize_rag_chain(llm, retriever, prompt):
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory}
    )
