# 📄 Chat with Your PDFs

A Streamlit web application that lets us upload a PDF document and chat with it using a powerful LLM (LLaMA 3 via Groq API). We can ask questions about the PDF content and get precise, concise answers in a conversational interface.

---

## Features

- Upload any PDF document via a simple web interface.
- The PDF is automatically processed, chunked, and embedded using HuggingFace embeddings.
- Use a RetrievalQA chain powered by the Groq-hosted LLaMA 3 model to answer questions based on the uploaded PDF.
- Maintains chat history during the session for a seamless conversational experience.
- User friendly interface with clear prompts and error handling.

---

## Used Technologies

- Streamlit
- LangChain
- Groq API (Provides access to the LLaMA 3 large language model)
- HuggingFace Embeddings (all-MiniLM-L12-v2)
- FAISS
- PyPDFLoader
- python-dotenv


## Screenshots

![image](https://github.com/user-attachments/assets/2112e748-5dd2-4287-9706-edb8242f3447)

---

![image](https://github.com/user-attachments/assets/dae98bee-b3aa-4bdc-93a9-fcfbf479cf12)

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- [Streamlit](https://streamlit.io/)
- Groq API key ([Groq](https://www.groq.com/))

### To Run

- Clone this repository and run below commands

   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt

   streamlit run groq_rag_chatbot.py
   ```
