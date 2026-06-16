import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="PDF Q&A Bot", layout="wide")
st.title("📄 PDF Question Answering Bot")
st.write("Upload a PDF and ask questions about it!")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    os.environ["OPENAI_API_KEY"] = api_key

# Main area
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.success("✓ PDF uploaded!")
    
    # Read PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Process
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
    chunks = splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    
    # Q&A
    st.subheader("Ask a Question")
    question = st.text_input("Your question:")
    
    if question:
        with st.spinner("Thinking..."):
            docs = retriever.invoke(question)
            context = docs[0].page_content[:800]
            response = llm.invoke(f"Based on this:\n{context}\n\nAnswer: {question}")
            st.write(response.content)
else:
    st.info("👆 Upload a PDF to get started!")