import warnings
import logging
import os

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Load API key
load_dotenv()

# Suppress warnings and logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title("Chatty")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

@st.cache_resource
def get_vectorstore():
    pdf_file = "Sequence alignment.pdf"
    if not os.path.exists(pdf_file):
        st.error(f"Missing PDF file: {pdf_file}")
        return None

    loaders = [PyPDFLoader(pdf_file)]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    ).from_loaders(loaders)
    return index.vectorstore

prompt = st.chat_input("Enter a message")

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    groq_sys_prompt = ChatPromptTemplate.from_template(
        """You are a very good AI assistant named "Chatty". You always provide precise and concise answers based on user input. Be nice and polite."""
    )

    model = "llama3-8b-8192"

    groq_chat = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name=model
    )

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.stop()

        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type='stuff',
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        result = chain({"query": prompt})
        response = result['result']

        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

    except Exception as e:
        st.error(f"Error: {e}")
