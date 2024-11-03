import streamlit as st
import os
import time
from constants import  *
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

groq_api_key=GROQ_API_KEY
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"

model_id = "sentence-transformers/all-MiniLM-L6-v2"

def vector_embedding(uploaded_file_path):
    if 'vectors' not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings(model_name=model_id)
        st.session_state.loader=PyMuPDFLoader(uploaded_file_path)
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.sidebar.title("Document Q&A Chatbot")

uploaded_file=st.sidebar.file_uploader("📂 Choose a pdf  file",type=['pdf'])

if uploaded_file is not None:
    with open("temp_uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.sidebar.button("Start Chat"):
        vector_embedding("temp_uploaded_file.pdf")
        st.sidebar.write("Your document is analyzed")

prompt1 = st.text_input('Enter the question')

llm = ChatGroq(
    api_key=groq_api_key,
    model='llama3-8b-8192')

prompt = ChatPromptTemplate([
    '''
    system, Answer the question based on the context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    ''',
    ' user, Question:{input}'
])

if prompt1:
    start = time.process_time()
    docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, docs_chain)
    response = retrieval_chain.invoke({'input': prompt1})
    st.sidebar.write('Response time :', time.process_time() - start)
    st.write(response.get('answer'))

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")