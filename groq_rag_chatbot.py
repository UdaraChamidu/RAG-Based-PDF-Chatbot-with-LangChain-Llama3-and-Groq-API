import warnings
import logging
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load API key
load_dotenv()

# Suppress warnings and logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title("📄 Chat with Your PDFs")

# Session state to preserve conversation and vectorstore
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load and index the PDF
    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
        chunks = splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
        vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
        st.session_state.vectorstore = vectorstore
        st.success("✅ PDF processed. You can now ask questions!")
    except Exception as e:
        st.error(f"Failed to process PDF: {e}")

# Display chat history
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Chat input
prompt = st.chat_input("Ask something about the PDF...")

if prompt and st.session_state.vectorstore:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # System prompt
    groq_sys_prompt = ChatPromptTemplate.from_template(
        """You are a very good AI assistant named "Chatty". You always provide precise and concise answers based on user input. Be nice and polite."""
    )

    # LLM Setup
    groq_chat = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama3-8b-8192"
    )

    try:
        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type='stuff',
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        result = chain({"query": prompt})
        response = result['result']

        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

    except Exception as e:
        st.error(f"Error: {e}")

elif prompt and not st.session_state.vectorstore:
    st.warning("Please upload a PDF first.")



